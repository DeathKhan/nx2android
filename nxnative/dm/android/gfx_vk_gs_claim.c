// Adreno executes a geometry shader separately for each declared output vertex. A memory-writing
// passthrough GS therefore needs one owner per input primitive, while all replicas still emit outputs.
// This conservative transform accepts only shaders whose suppressed atomic results cannot affect
// raster outputs or control flow. Storage-only void helpers are guarded as a whole: guarding their
// individual compare-exchanges would leave losing replicas spinning in a subword-store retry loop.
#include "gfx_vk_gs_claim.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct { uint32_t at,n,op,fn,id; } ins;
typedef struct { uint32_t type,root,sc,op,next,entry,ret,params,last; unsigned istype:1,param:1,taint:1,memtaint:1,effect:1,helper:1,returns_taint:1; } ident;
typedef struct { uint32_t *p,n,cap; } vec;
static int append(vec *v,const uint32_t *p,uint32_t n) { if(!n)return 1; if(n>UINT32_MAX-v->n)return 0; if(v->n+n>v->cap){uint32_t c=(v->n+n)*2;void*q=realloc(v->p,(size_t)c*4);if(!q)return 0;v->p=q;v->cap=c;}memcpy(v->p+v->n,p,(size_t)n*4);v->n+=n;return 1; }
static int emit(vec*v,uint32_t op,const uint32_t*a,uint32_t n){uint32_t h=((n+1)<<16)|op;return append(v,&h,1)&&append(v,a,n);}
#define E(v,op,...) do { uint32_t a_[]={__VA_ARGS__}; if(!emit(v,op,a_,sizeof(a_)/4))goto fail; } while(0)
#define Z(v,op) do { if(!emit(v,op,NULL,0))goto fail; } while(0)
static int external(uint32_t sc){return sc==2||sc==12||sc==5349;}
static int atomic_op(uint32_t op){return op>=227&&op<=242&&op!=228;}
static int effect_op(const ins*i,const uint32_t*w,const ident*d,uint32_t bound){
 const uint32_t*x=w+i->at;uint32_t p=0;
 if(i->op==62)p=x[1];else if(atomic_op(i->op))p=x[3];else if(i->op==228)p=x[1];else if(i->op==99)return 1;
 return p<bound&&p&&external(d[p].sc);
}
static uint32_t pointer_type(const ins*in,uint32_t ni,const uint32_t*w,uint32_t sc,uint32_t elem){
 for(uint32_t i=0;i<ni;i++){const uint32_t*x=w+in[i].at;if(in[i].op==32&&in[i].n==4&&x[2]==sc&&x[3]==elem)return x[1];}return 0;
}
static uint32_t constant(const ins*in,uint32_t ni,const uint32_t*w,uint32_t t,uint32_t value){
 for(uint32_t i=0;i<ni;i++){const uint32_t*x=w+in[i].at;if(in[i].op==43&&in[i].n==4&&x[1]==t&&x[3]==value)return x[2];}return 0;
}
static uint32_t function_type(const ins*in,uint32_t ni,const uint32_t*w,const vec*decl,const uint32_t*types,uint32_t n){
 for(uint32_t i=0;i<ni;i++){const uint32_t*x=w+in[i].at;if(in[i].op==33&&in[i].n==n+2&&!memcmp(x+2,types,n*4))return x[1];}
 for(uint32_t i=0;i<decl->n;){const uint32_t*x=decl->p+i;uint32_t z=x[0]>>16;if((x[0]&65535)==33&&z==n+2&&!memcmp(x+2,types,n*4))return x[1];i+=z;}return 0;
}
uint32_t *spv_gs_claim(const uint32_t*w,uint32_t nw,uint32_t set,uint32_t binding,uint32_t*out_words,uint32_t*invocations,char*why,size_t wsz){
 ins*in=NULL;ident*d=NULL;uint32_t*wrap=NULL,*keep=NULL;vec dec={0},decl={0},funcs={0},out={0};const char*reason="allocation failed";
 uint32_t ni=0,entries=0,main=0,fn=0,u32=0,bt=0,vt=0,prim=0,iv=1,tri=0,bound=0,next=0;
 if(out_words)*out_words=0;if(invocations)*invocations=0;if(why&&wsz)*why=0;
 if(!w||nw<5||nw>(1u<<24)||w[0]!=0x07230203||w[3]<1||w[3]>(1u<<20)){reason="invalid module";goto fail;}
 bound=w[3];next=bound;in=calloc(nw,sizeof *in);d=calloc(bound,sizeof *d);wrap=calloc(nw,sizeof *wrap);keep=calloc(bound,sizeof *keep);if(!in||!d||!wrap||!keep)goto fail;
 for(uint32_t at=5;at<nw;){uint32_t n=w[at]>>16,op=w[at]&65535;const uint32_t*x=w+at;if(!n||n>nw-at){reason="invalid instruction";goto fail;}
  uint32_t need=1;
  switch(op){case 15:need=4;break;case 16:case 71:need=3;break;case 72:need=4;break;case 54:need=5;break;case 55:need=3;break;case 57:case 59:case 61:case 83:need=4;break;case 62:need=3;break;case 65:case 66:case 67:case 68:case 70:need=5;break;case 245:need=5;break;case 248:case 249:case 254:need=2;break;case 250:need=4;break;case 251:need=3;break;}
  if(n<need){reason="truncated instruction";goto fail;}
  if(op>=227&&op<=242){uint32_t expected=op==227?6:op==228?5:op==230?9:(op==232||op==233)?6:7;if(n!=expected||op==231){reason="unsupported atomic shape";goto fail;}for(uint32_t j=1;j<n;j++)if(x[j]>=bound){reason="atomic ID outside bound";goto fail;}}
  if((op==54||op==55||op==57||op==59||op==61||op==65||op==66||op==67||op==68||op==70||op==83||op==245)&&(x[1]>=bound||x[2]>=bound)){reason="result ID outside bound";goto fail;}
  if((op==15&&x[2]>=bound)||(op==248&&x[1]>=bound)||(op==62&&(x[1]>=bound||x[2]>=bound))){reason="instruction ID outside bound";goto fail;}
  if(op==57||op==61||op==65||op==66||op==67||op==70||op==83||op==245)for(uint32_t j=3;j<n;j++)if(x[j]>=bound){reason="operand ID outside bound";goto fail;}
  if(op==59&&n>=5&&x[4]>=bound){reason="initializer ID outside bound";goto fail;}
  if(op==15&&++entries>1){reason="multiple entry points";goto fail;}
  if(op==15&&n>=4&&x[1]==3){if(main){reason="multiple geometry entries";goto fail;}main=x[2];}
  if(op==71&&n>=4&&x[2]==33&&x[3]==binding){reason="claim binding occupied";goto fail;}
  if(op==71&&n>=3&&x[1]<bound&&x[2]==3)d[x[1]].ret=3; // BufferBlock
  if(op==73||op==74||op==75){reason="decoration groups unsupported";goto fail;}
  if(op>=19&&op<=39&&n>=2&&x[1]<bound){d[x[1]].istype=1;d[x[1]].op=op;if(op==21&&n==4&&x[2]==32&&!x[3])u32=x[1];if(op==20)bt=x[1];if(op==19)vt=x[1];if(op==32&&n==4){d[x[1]].sc=x[2];d[x[1]].type=x[3];}if(op==21&&n==4)d[x[1]].ret=x[2];}
  uint32_t id=0;if(n>=3&&x[1]<bound&&d[x[1]].istype&&x[2]<bound&&op>=41){id=x[2];d[id].type=x[1];d[id].op=op;d[id].sc=d[x[1]].sc;}
  if(op==54){fn=x[2];d[fn].ret=x[1];d[fn].helper=1;}if(op==248&&fn&&!d[fn].entry)d[fn].entry=x[1];
  if(op==59&&n>=4&&id){d[id].root=id;d[id].sc=x[3];}if(op==55&&id){d[id].root=id;d[id].param=1;if(d[fn].last)d[d[fn].last].next=id;else d[fn].params=id;d[fn].last=id;}
  if((op==65||op==66||op==67||op==70||op==83)&&n>=4&&id&&x[3]<bound)d[id].root=d[x[3]].root;
  in[ni++]=(ins){at,n,op,fn,id};if(op==56)fn=0;at+=n;
 }
 if(!main||main>=bound||d[main].op!=54||!u32||!bt||!vt){reason="not a supported geometry module";goto fail;}
 for(uint32_t i=0;i<ni;i++){const ins*q=&in[i];const uint32_t*x=w+q->at;
  if(q->op==16&&q->n>=3&&x[1]==main){if(x[2]==0&&q->n==4)iv=x[3];if(x[2]==22)tri=1;}
  if(q->op==71&&q->n==4&&x[2]==11&&x[3]==7&&x[1]<bound&&d[x[1]].sc==1)prim=x[1];
  if(q->op==72&&q->n==5&&x[3]==11&&x[4]==7){reason="member-decorated primitive ID unsupported";goto fail;}
 }
 // Reject pointer aliases and instruction families whose memory/raster effects are not modeled.
 for(uint32_t i=0;i<ni;i++){const ins*q=&in[i];const uint32_t*x=w+q->at;if(!q->fn)continue;uint32_t op=q->op;
  if(op==12){int math=0;if(q->n<5){reason="truncated extended instruction";goto fail;}for(uint32_t j=0;j<ni;j++)if(in[j].op==11&&in[j].n>=5&&w[in[j].at+1]==x[3]&&!memcmp(w+in[j].at+2,"GLSL.std.450",12))math=1;if(!math){reason="unknown extended instruction set";goto fail;}for(uint32_t j=5;j<q->n;j++)if(x[j]<bound&&d[x[j]].type<bound&&d[d[x[j]].type].op==32){reason="extended instruction pointer escape";goto fail;}}
  if(!(op==0||op==8||op==12||(op>=54&&op<=57)||op==59||op==61||op==62||op==65||op==66||op==67||op==68||op==70||(op>=77&&op<=107)||(op>=109&&op<=215)||(op>=218&&op<=221)||(op>=227&&op<=242)||(op>=245&&op<=255)||op==317)){reason="unmodeled instruction family";goto fail;}
  if(q->id&&d[d[q->id].type].op==32){
   if(!(op==59||op==55||op==65||op==66||op==67||op==70||op==83)||!d[q->id].root){reason="unproven pointer alias";goto fail;}
  }
  if(op==54&&d[x[1]].op==32){reason="pointer-returning function";goto fail;}
 }
 // Determine which void helpers can be bypassed without changing their caller's local state or stage outputs.
 for(uint32_t i=0;i<ni;i++){const ins*q=&in[i];const uint32_t*x=w+q->at;if(!q->fn)continue;ident*f=&d[q->fn];
  if(q->op==99){reason="image side effects unsupported";goto fail;}
  if(f->ret!=vt||q->op==254||(q->op>=218&&q->op<=221)||q->op==252)f->helper=0;
  if(q->op==62){uint32_t p=x[1],r=p<bound?d[p].root:0,sc=p<bound?d[p].sc:0;
   if(!r||d[r].param||(!external(sc)&&sc!=7))f->helper=0;
  }
  if(q->op==63||q->op==64||q->op==224||q->op==225){reason="memory copy or barrier unsupported";goto fail;}
  if(effect_op(q,w,d,bound)){uint32_t p=q->op==62||q->op==228?x[1]:x[3],root=d[p].root;
   if(!root||d[root].op!=59){reason="external effect through unproven global root";goto fail;}f->effect=1;}

 }
 for(unsigned pass=0;pass<bound;pass++){int changed=0;
  for(uint32_t i=0;i<ni;i++){const ins*q=&in[i];const uint32_t*x=w+q->at;if(q->op!=57||!q->fn||q->n<4)continue;uint32_t c=x[3];if(c>=bound){reason="invalid call";goto fail;}
   if(d[c].effect&&!d[q->fn].effect){d[q->fn].effect=1;changed=1;}if(!d[c].helper&&d[q->fn].helper){d[q->fn].helper=0;changed=1;}
  }if(!changed)break;
 }
 if(!d[main].effect){reason="no memory side effects";goto fail;}
 if(iv!=1||!tri){reason="only single-invocation triangle GS supported";goto fail;}
 d[main].helper=0;
 for(uint32_t i=0;i<ni;i++){const ins*q=&in[i];if(q->fn&&q->fn!=main&&d[q->fn].effect&&!d[q->fn].helper){reason="side-effect helper changes caller state";goto fail;}}
 for(uint32_t i=0;i<ni;i++){const ins*q=&in[i];const uint32_t*x=w+q->at;if(!q->fn||q->fn==main||!d[q->fn].effect)continue;uint32_t e=d[q->fn].entry;
  if((q->op==249&&x[1]==e)||(q->op==250&&(x[2]==e||x[3]==e))){reason="helper entry is a loop target";goto fail;}
 }
 // Atomic values omitted in replicas must not influence emitted outputs or branch decisions.
 // Storage bindings may alias. Any storage read used by raster must remain unchanged on every replica.
 for(uint32_t id=1;id<bound;id++)if(d[id].op==59){uint32_t pt=d[id].type,st=pt<bound?d[pt].type:0;
  if(d[id].sc==12||d[id].sc==5349||(d[id].sc==2&&st<bound&&d[st].ret==3))d[id].memtaint=1;}
 for(uint32_t i=0;i<ni;i++)if(in[i].fn==main&&atomic_op(in[i].op)&&effect_op(&in[i],w,d,bound))d[in[i].id].taint=1;
 for(unsigned pass=0;pass<bound;pass++){int changed=0;
  for(uint32_t i=0;i<ni;i++){const ins*q=&in[i];const uint32_t*x=w+q->at;if(!q->fn||d[q->fn].helper)continue;
   if(q->op==57&&q->n>=4&&d[x[3]].helper)continue;
   int taint=0;uint32_t begin=q->id?3:1;
   for(uint32_t j=begin;j<q->n;j++)if(x[j]<bound&&d[x[j]].taint)taint=1;
   if(q->op==61&&q->n>=4&&x[3]<bound){uint32_t r=d[x[3]].root;if(r&&d[r].memtaint)taint=1;}
   if(q->op==57&&q->n>=4){uint32_t p=d[x[3]].params;unsigned j=4;
    while(p&&j<q->n){uint32_t a=x[j++],r=a<bound?d[a].root:0;
     if((a<bound&&d[a].taint)||(r&&d[r].memtaint)){if(!d[p].taint||!d[p].memtaint){d[p].taint=d[p].memtaint=1;changed=1;}}
     if(r&&d[p].memtaint&&!d[r].memtaint){d[r].memtaint=1;changed=1;}p=d[p].next;
    }if(d[x[3]].returns_taint)taint=1;
   }
   if(q->id&&taint&&!d[q->id].taint){d[q->id].taint=1;changed=1;}
   if(q->op==62&&q->n>=3&&x[1]<bound){uint32_t r=d[x[1]].root;
    if(taint&&r&&!d[r].memtaint){d[r].memtaint=1;changed=1;}
    if(taint&&d[x[1]].sc==3){reason="atomic result reaches raster output";goto fail;}
   }
   if((q->op==250||q->op==251)&&q->n>=2&&x[1]<bound&&d[x[1]].taint){reason="atomic result reaches control flow";goto fail;}
   if(q->op==254&&taint&&!d[q->fn].returns_taint){d[q->fn].returns_taint=1;changed=1;}
  }if(!changed)break;
 }
 uint32_t c0=constant(in,ni,w,u32,0),c1=constant(in,ni,w,u32,1);if(!c0){c0=next++;E(&decl,43,u32,c0,0);}if(!c1){c1=next++;E(&decl,43,u32,c1,1);}
 uint32_t pbool=pointer_type(in,ni,w,6,bt);if(!pbool){pbool=next++;E(&decl,32,pbool,6,bt);}uint32_t owner=next++;E(&decl,59,pbool,owner,6);
 uint32_t pin=pointer_type(in,ni,w,1,u32);if(!pin){pin=next++;E(&decl,32,pin,1,u32);}int added_prim=!prim;
 if(!prim){prim=next++;E(&decl,59,pin,prim,1);E(&dec,71,prim,11,7);}else if(d[prim].type>=bound||d[d[prim].type].op!=32){reason="invalid primitive input";goto fail;}
 uint32_t arr=next++,block=next++,pblock=next++,pu=next++,buffer=next++;E(&decl,29,arr,u32);E(&dec,71,arr,6,4);E(&decl,30,block,u32,arr);E(&dec,71,block,2);E(&dec,72,block,0,35,0);E(&dec,72,block,1,35,4);
 E(&decl,32,pblock,12,block);uint32_t oldpu=pointer_type(in,ni,w,12,u32);if(oldpu)pu=oldpu;else E(&decl,32,pu,12,u32);E(&decl,59,pblock,buffer,12);E(&dec,71,buffer,34,set);E(&dec,71,buffer,33,binding);
 uint32_t ft=next++,claim=next++,entry=next++,doit=next++,fallback=next++;
 uint32_t oldft=function_type(in,ni,w,&decl,&bt,1);if(oldft)ft=oldft;else E(&decl,33,ft,bt);uint32_t yes=next++;E(&decl,41,bt,yes);
 E(&funcs,54,bt,claim,0,ft);E(&funcs,248,entry);
 uint32_t ep=next++,enabled=next++,active=next++;E(&funcs,65,pu,ep,buffer,c0);E(&funcs,61,u32,enabled,ep);E(&funcs,171,bt,active,enabled,c0);
 uint32_t pid=next++,len=next++,inside=next++,run=next++;
 uint32_t primtype=added_prim?u32:d[d[prim].type].type;if(primtype>=bound&&!added_prim){reason="invalid primitive scalar";goto fail;}
 if(!added_prim&&(d[primtype].op!=21||d[primtype].ret!=32)){reason="unsupported primitive scalar";goto fail;}
 if(primtype==u32)E(&funcs,61,u32,pid,prim);else {uint32_t raw=next++;E(&funcs,61,primtype,raw,prim);E(&funcs,124,u32,pid,raw);}E(&funcs,68,u32,len,buffer,1);E(&funcs,176,bt,inside,pid,len);E(&funcs,167,bt,run,active,inside);
 E(&funcs,247,fallback,0);E(&funcs,250,run,doit,fallback);E(&funcs,248,doit);
 uint32_t cp=next++,old=next++,first=next++;E(&funcs,65,pu,cp,buffer,c1,pid);E(&funcs,229,u32,old,cp,c1,c0,c1);E(&funcs,170,bt,first,old,c0);E(&funcs,254,first);
 E(&funcs,248,fallback);E(&funcs,254,yes);Z(&funcs,56);
 // Guard individual main operations using scalar operands and the original global storage object.
 for(uint32_t i=0;i<ni;i++){const ins*q=&in[i];const uint32_t*x=w+q->at;if(q->fn!=main||!effect_op(q,w,d,bound))continue;
  if(q->op!=62&&q->op!=228&&!atomic_op(q->op)){reason="unsupported direct memory side effect";goto fail;}
  uint32_t rt=q->id?x[1]:vt,opid=q->id?x[2]:0,np=0,pos[16],where[16],pt[16],pv[16],orig[16];
  uint32_t pp=q->op==62||q->op==228?1:3,pid=x[pp];const uint32_t *px=NULL;uint32_t pn=0;
  for(uint32_t j=0;j<ni;j++)if(in[j].id==pid&&(in[j].op==65||in[j].op==66)){px=w+in[j].at;pn=in[j].n;break;}
  if(!px||pn>16||px[3]>=bound||d[px[3]].op!=59){reason="side-effect pointer is not a direct global access chain";goto fail;}
  for(uint32_t j=4;j<pn;j++)if(px[j]>=bound||d[px[j]].op<41||d[px[j]].op>52){where[np]=0;pos[np++]=j;}
  if(q->op==62){where[np]=1;pos[np++]=2;}else if(q->op==228){where[np]=1;pos[np++]=4;}else for(uint32_t j=q->op==230?7:6;j<q->n;j++){where[np]=1;pos[np++]=j;}
  if(np>16){reason="unsupported atomic operand count";goto fail;}
  for(uint32_t j=0;j<np;j++){orig[j]=where[j]?x[pos[j]]:px[pos[j]];if(orig[j]>=bound||!d[orig[j]].type){reason="unknown side-effect operand type";goto fail;}pt[j]=d[orig[j]].type;pv[j]=next++;}
  uint32_t wf=next++,wid=next++,we=next++,wt=next++,wm=next++,cond=next++;wrap[i]=wid;
  uint32_t ts[18]={wf,rt};for(uint32_t j=0;j<np;j++)ts[2+j]=pt[j];uint32_t oldwf=function_type(in,ni,w,&decl,ts+1,np+1);if(oldwf)wf=oldwf;else if(!emit(&decl,33,ts,np+2))goto fail;
  E(&funcs,54,rt,wid,0,wf);for(uint32_t j=0;j<np;j++)E(&funcs,55,pt[j],pv[j]);E(&funcs,248,we);E(&funcs,61,bt,cond,owner);E(&funcs,247,wm,0);E(&funcs,250,cond,wt,wm);E(&funcs,248,wt);
  uint32_t clone[20],pcopy[16];if(q->n>20){reason="oversized side effect";goto fail;}memcpy(clone,x,q->n*4);memcpy(pcopy,px,pn*4);pcopy[2]=next++;clone[pp]=pcopy[2];uint32_t val=0;if(opid)clone[2]=val=next++;
  for(uint32_t j=0;j<np;j++)if(where[j])clone[pos[j]]=pv[j];else pcopy[pos[j]]=pv[j];
  if(!append(&funcs,pcopy,pn)||!append(&funcs,clone,q->n))goto fail;
  E(&funcs,249,wm);E(&funcs,248,wm);if(opid){uint32_t zero=next++,result=next++;E(&decl,46,rt,zero);E(&funcs,245,rt,result,val,wt,zero,we);E(&funcs,254,result);}else Z(&funcs,253);Z(&funcs,56);
 }
 for(uint32_t i=0;i<ni;i++)if(in[i].op==54&&in[i].id!=main&&d[in[i].id].helper&&d[in[i].id].effect)keep[in[i].id]=next++;
 if(!append(&out,w,5))goto fail;int dec_done=0,decl_done=0;uint32_t current=0,entry_label=0;int prologue=0;
 for(uint32_t i=0;i<ni;i++){const ins*q=&in[i];const uint32_t*x=w+q->at;
  if(!dec_done&&q->op>=19&&q->op<=39){if(!append(&out,dec.p,dec.n))goto fail;dec_done=1;}
  if(!decl_done&&q->op==54){if(!append(&out,decl.p,decl.n))goto fail;decl_done=1;}
  if(q->op==54){current=x[2];entry_label=d[current].entry;prologue=current==main||keep[current];}
  if(prologue&&q->fn==current&&q->op!=54&&q->op!=55&&q->op!=248&&q->op!=59&&q->op!=8&&q->op!=317){
   if(current==main){uint32_t v=next++;E(&out,57,bt,v,claim);E(&out,62,owner,v);}
   else {uint32_t v=next++,exit=next++;E(&out,61,bt,v,owner);E(&out,247,keep[current],0);E(&out,250,v,keep[current],exit);E(&out,248,exit);Z(&out,253);E(&out,248,keep[current]);}
   prologue=0;
  }
  if(q->op==15&&x[1]==3&&(added_prim||w[1]>=0x10400)){uint32_t extra[3],ne=0;if(added_prim)extra[ne++]=prim;if(w[1]>=0x10400){extra[ne++]=owner;extra[ne++]=buffer;}uint32_t h=x[0]+(ne<<16);if(!append(&out,&h,1)||!append(&out,x+1,q->n-1)||!append(&out,extra,ne))goto fail;continue;}
  if(wrap[i]){uint32_t a[32],n=0;a[n++]=q->id?x[1]:vt;a[n++]=q->id?x[2]:next++;a[n++]=wrap[i];uint32_t pid=x[q->op==62||q->op==228?1:3];
   for(uint32_t j=0;j<ni;j++)if(in[j].id==pid&&(in[j].op==65||in[j].op==66)){const uint32_t*px=w+in[j].at;for(uint32_t k=4;k<in[j].n;k++)if(d[px[k]].op<41||d[px[k]].op>52)a[n++]=px[k];break;}
   if(q->op==62)a[n++]=x[2];else if(q->op==228)a[n++]=x[4];else for(uint32_t j=q->op==230?7:6;j<q->n;j++)a[n++]=x[j];if(!emit(&out,57,a,n))goto fail;continue;}
  if(q->op==245&&keep[current]){uint32_t*copy=malloc(q->n*4);if(!copy)goto fail;memcpy(copy,x,q->n*4);for(uint32_t j=4;j<q->n;j+=2)if(copy[j]==entry_label)copy[j]=keep[current];int ok=append(&out,copy,q->n);free(copy);if(!ok)goto fail;continue;}
  if(!append(&out,x,q->n))goto fail;
 }
 if(!append(&out,funcs.p,funcs.n))goto fail;out.p[3]=next;if(out_words)*out_words=out.n;if(invocations)*invocations=1;
 free(in);free(d);free(wrap);free(keep);free(dec.p);free(decl.p);free(funcs.p);if(why&&wsz)snprintf(why,wsz,"primitive ownership, storage-only side effects");return out.p;
fail:
 if(why&&wsz)snprintf(why,wsz,"%s",reason);free(in);free(d);free(wrap);free(keep);free(dec.p);free(decl.p);free(funcs.p);free(out.p);return NULL;
}

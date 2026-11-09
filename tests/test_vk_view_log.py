"""Passive native-view logger: strict bounds, cache-hit coverage and no resource calls."""
from pathlib import Path
import shutil,subprocess,tempfile,unittest
from test_vk_storage_image import function
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'nxnative/dm/android/gfx_vk_tex.c'
class ViewLog(unittest.TestCase):
 @unittest.skipUnless(shutil.which('cc'),'host compiler required')
 def test_actual_helper_filter_ranges_and_budget(self):
  s=SRC.read_text();helpers='\n'.join(function(s,n) for n in ['static int view_log_size_parse(','static void view_log_selected('])
  code=r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
typedef uintptr_t VkImageView;
typedef struct {uint64_t serial,off;void *pool;struct{int target;}d;}tex_t;
typedef struct {int set_format,set_target,set_levels,set_layers,set_swizzle,set_dsmode,target,format;uint32_t base_level,nlevels,min_layer,nlayers;}view_t;
typedef struct {void *img;uint32_t w,h,d,fmt,vtype,layers,levels,samples;int target;}vkimg_t;
typedef struct {VkImageView v;uint32_t type,fmt,lv0,nlv,ly0,nly,aspect;}vview;
static unsigned logs;static char last[4096];
static void note(const char *fmt,...){va_list ap;va_start(ap,fmt);vsnprintf(last,sizeof last,fmt,ap);va_end(ap);logs++;}
#define VKLOG(...) note(__VA_ARGS__)
HELPERS
int main(int argc,char **argv){
 assert(argc==2);uint32_t w=77,h=88;
 const char *bad[]={NULL,"","0x928","1524x0","-1x928","1524X928","1524x928tail","1524","1524xx928","4294967296x928","1524x4294967296"," 1524x928","1524x928 "};
 for(unsigned i=0;i<sizeof bad/sizeof *bad;i++){assert(!view_log_size_parse(bad[i],&w,&h));assert(w==77&&h==88);}
 assert(view_log_size_parse("1524x928",&w,&h)&&w==1524&&h==928);
 assert(view_log_size_parse("4294967295x1",&w,&h)&&w==UINT32_MAX&&h==1);
 if(strcmp(argv[1],"off"))setenv("DM_VK_VIEW_LOG_SIZE",argv[1],1);else unsetenv("DM_VK_VIEW_LOG_SIZE");
 tex_t t={.serial=123,.off=456,.pool=(void*)9,.d={4}},saved=t;view_t v={.set_layers=1,.nlayers=3};
 vkimg_t im={.img=(void*)99,.w=1524,.h=928,.d=1,.fmt=143,.vtype=5,.layers=3,.levels=1,.samples=1,.target=4};
 vview n={.v=1,.type=1,.fmt=143,.nlv=1,.nly=1,.aspect=1};
 view_log_selected(&t,&v,&im,1,0,0,&n,1);
 if(strcmp(argv[1],"1524x928")){assert(!logs);return 0;}
 assert(logs==1&&strstr(last,"cached=1")&&strstr(last,"request_arrayed=0")&&strstr(last,"layers=0+1"));
 for(int i=0;i<100;i++)view_log_selected(&t,&v,&im,1,0,0,&n,1);
 assert(logs==1);n.v=2;n.type=5;n.nly=3;view_log_selected(&t,&v,&im,1,1,0,&n,0);
 assert(logs==2&&strstr(last,"request_arrayed=1")&&strstr(last,"type=5")&&strstr(last,"layers=0+3")&&strstr(last,"native_layers=3"));
 im.w=1;n.v=3;view_log_selected(&t,&v,&im,1,1,0,&n,1);assert(logs==2);im.w=1524;
 for(unsigned i=3;i<100;i++){n.v=i;view_log_selected(&t,&v,&im,1,1,0,&n,1);}
 assert(logs==16&&!memcmp(&t,&saved,sizeof t));return 0;
}
'''.replace('HELPERS',helpers)
  target=ROOT/'target/tests';target.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='view-log-',dir=target)as tmp:
   c,e=Path(tmp)/'t.c',Path(tmp)/'t';c.write_text(code)
   subprocess.run(['cc','-std=gnu11','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',str(c),'-o',str(e)],check=True)
   for value in ['off','1524x928','1524x928bad','4294967296x928']:subprocess.run([str(e),value],check=True,timeout=10)
 def test_actual_cached_and_created_hooks(self):
  body=function(SRC.read_text(),'VkImageView vk_tex_view(')
  self.assertEqual(body.count('view_log_selected('),2)
  self.assertIn('view_log_selected(t,vw,im,dim,arrayed,view_kind,v,1); pthread_mutex_unlock',body)
  self.assertIn('view_log_selected(t,vw,im,dim,arrayed,view_kind,nv,0);\n    pthread_mutex_unlock',body)
if __name__=='__main__':unittest.main()

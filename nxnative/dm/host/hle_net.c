// Networking (out of scope: online play/friends/store/cloud saves): all hooked up as "network unavailable", the game takes the offline branch.
// Cuts: ssl and friends are hooked up as failures when they come into use. nn::socket is in hle_socket.c, nifm in hle_ldn.c (both offline unless local play over the LAN is on).
#include "hle.h"
#include <pthread.h>
#include <stdatomic.h>

// TODO(upsync): upstream (seedds/nxnative main) added to the socket / nifm hooks that this fork moved to hle_socket.c and hle_ldn.c; port there (upstream's hle_net.c has the code):
//  - nn::socket::internal::Initialize(int) (SDK 21 inlines Initialize(const Config &) into the game, Star Fox): succeed, no session
//  - nifm: SubmitNetworkRequest leaves the request "on hold" for the first IsNetworkRequestOnHold query after the submission, then settled (Pikmin 4's BCAT thread; Hades' poller):
//    `static _Atomic int g_nifm_hold`, Submit stores 1, Cancel 0, OnHold decrements and answers h > 0
//  - HandleNetworkRequestErrorResult returns bool (`and w0, w0, #1`, Metroid Dread): 0 = not resolved, not a Result
HLE_FORCE("_ZN2nn4nifm22GetCurrentProxySettingEPNS0_12ProxySettingE", "offline: no proxy setting", Result, h_nifm_GetProxy, void *p) { (void)p; return RES_FAIL; }
HLE_FORCE("nnsslInitialize", "connects to the ssl service; the network library's global init needs it to succeed, offline creates no session (cut)", Result, h_ssl_Initialize, void) { return RES_OK; }
HLE_FORCE("nnsslFinalize", "pairs with nnsslInitialize (cut)", Result, h_ssl_Finalize, void) { return RES_OK; }

// The offline socket layer lives in hle_socket.c here (with local play over the LAN); upstream's hle_net.c socket section (EventFd host counters, Poll of descriptors the
// host never made, Select as Poll, event descriptors numbered from 0, the SDK 21 overloads and the C library's poll) is ported there (upsync of eddae74f3)

// ---------- ssl: offline, Initialize and a context's Create / Destroy succeed, everything else that contacts the service fails ----------
// A context is a session of the local ssl service, so on Horizon Create works with no network and only a connection needs one; DRAGON QUEST BUILDERS 2 ends its startup when Create fails (main+0x861260).
// Not cut: Context / Connection construction and destruction, Context::GetContextId (only read and write game objects).
#define SSF(m, cname, ...) HLE_FORCE(m, "ssl service (cut): fails", Result, cname, __VA_ARGS__)
HLE_FORCE("_ZN2nn3ssl10InitializeEv", "same as nnsslInitialize: succeeds, creates no session (cut)", Result, h_ssl_InitCpp, void) { return RES_OK; }
HLE_FORCE("_ZN2nn3ssl8FinalizeEv", "pairs with Initialize (cut)", Result, h_ssl_FinCpp, void) { return RES_OK; }
HLE_FORCE("_ZN2nn3ssl7Context6CreateENS1_10SslVersionE", "a session of the local ssl service: succeeds offline, connects to nothing (cut)", Result, h_ssl_CtxCreate, void *self, int v) { (void)self; (void)v; return RES_OK; }
HLE_FORCE("_ZN2nn3ssl7Context7DestroyEv", "pairs with Create (cut)", Result, h_ssl_CtxDestroy, void *self) { (void)self; return RES_OK; }
SSF("_ZN2nn3ssl7Context19RegisterInternalPkiEPmNS1_11InternalPkiE", h_ssl_CtxRegPki, void *self, uint64_t *id, int p) { (void)self; (void)id; (void)p; return RES_FAIL; }
SSF("_ZN2nn3ssl7Context9RemovePkiEm", h_ssl_CtxRemovePki, void *self, uint64_t id) { (void)self; (void)id; return RES_FAIL; }
SSF("_ZN2nn3ssl14BuiltInManager28GetBuiltInCertificateBufSizeEPjPNS0_15CaCertificateIdEj", h_ssl_BiBufSize, uint32_t *o, void *ids, unsigned n) { (void)ids; (void)n; if (o) *o = 0; return RES_FAIL; }
SSF("_ZN2nn3ssl14BuiltInManager22GetBuiltInCertificatesEPPNS1_22BuiltInCertificateInfoEPhjPNS0_15CaCertificateIdEj", h_ssl_BiCerts, void **o, uint8_t *b, unsigned bs, void *ids, unsigned n) { (void)o; (void)b; (void)bs; (void)ids; (void)n; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection6CreateEPNS0_7ContextE", h_ssl_CnCreate, void *self, void *ctx) { (void)self; (void)ctx; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection7DestroyEv", h_ssl_CnDestroy, void *self) { (void)self; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection11DoHandshakeEv", h_ssl_CnHandshake, void *self) { (void)self; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection11SetHostNameEPKcj", h_ssl_CnHost, void *self, const char *h, unsigned n) { (void)self; (void)h; (void)n; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection9SetIoModeENS1_6IoModeE", h_ssl_CnIoMode, void *self, int m) { (void)self; (void)m; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection19SetSessionCacheModeENS1_16SessionCacheModeE", h_ssl_CnCache, void *self, int m) { (void)self; (void)m; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection19SetSocketDescriptorEi", h_ssl_CnSock, void *self, int s) { (void)self; (void)s; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection12GetLastErrorEPNS_6ResultE", h_ssl_CnLastErr, void *self, uint32_t *r) { (void)self; if (r) *r = RES_FAIL; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection18GetVerifyCertErrorEPNS_6ResultE", h_ssl_CnVerifyErr, void *self, uint32_t *r) { (void)self; if (r) *r = RES_FAIL; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection4PeekEPcPij", h_ssl_CnPeek, void *self, char *b, int *o, unsigned n) { (void)self; (void)b; (void)n; if (o) *o = 0; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection4PollEPNS1_9PollEventES3_j", h_ssl_CnPoll, void *self, void *o, void *i, unsigned t) { (void)self; (void)o; (void)i; (void)t; return RES_FAIL; }
HLE_FORCE("_ZN2nn3ssl10Connection4ReadEPcj", "ssl service (cut): read returns -1", int, h_ssl_CnRead, void *self, char *b, unsigned n) { (void)self; (void)b; (void)n; return -1; }
HLE_FORCE("_ZN2nn3ssl10Connection5WriteEPKcj", "ssl service (cut): write returns -1", int, h_ssl_CnWrite, void *self, const char *b, unsigned n) { (void)self; (void)b; (void)n; return -1; }
// C interface: all return nnResult
#define SSC(m, cname) HLE_FORCE(m, "ssl service (cut): fails", Result, cname, void) { return RES_FAIL; }
SSC("nnsslConnectionCreate", h_nssl_Create) SSC("nnsslConnectionDestroy", h_nssl_Destroy)
SSC("nnsslConnectionDoHandshakeWithCertBuffer", h_nssl_Handshake) SSC("nnsslConnectionGetConnectionId", h_nssl_GetId)
SSC("nnsslConnectionGetNextAlpnProto", h_nssl_GetAlpn) SSC("nnsslConnectionGetVerifyCertError", h_nssl_VerifyErr)
SSC("nnsslConnectionPeek", h_nssl_Peek) SSC("nnsslConnectionPending", h_nssl_Pending) SSC("nnsslConnectionRead", h_nssl_Read)
SSC("nnsslConnectionSetHostName", h_nssl_SetHost) SSC("nnsslConnectionSetIoMode", h_nssl_SetIo)
SSC("nnsslConnectionSetNextAlpnProto", h_nssl_SetAlpn) SSC("nnsslConnectionSetOption", h_nssl_SetOpt)
SSC("nnsslConnectionSetSessionCacheMode", h_nssl_SetCache) SSC("nnsslConnectionSetSocketDescriptor", h_nssl_SetSock)
SSC("nnsslConnectionSetVerifyOption", h_nssl_SetVerify) SSC("nnsslConnectionWrite", h_nssl_Write)
SSC("nnsslContextDestroy", h_nssl_CtxDestroy) SSC("nnsslContextGetContextId", h_nssl_CtxGetId)

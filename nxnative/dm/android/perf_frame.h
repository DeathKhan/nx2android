#ifndef DM_PERF_FRAME_H
#define DM_PERF_FRAME_H
#include <stdint.h>
#include <string.h>
#include <pthread.h>

#define PERF_FRAME_COUNT 64
typedef struct {
    uint64_t id, start, end, replay, pace, pq;
    int64_t work, cpu, gpu_estimate;
    unsigned present:1, closed:1, reported:1, invalid:1;
} perf_frame_record;
typedef struct {
    pthread_mutex_t mutex;
    perf_frame_record frames[PERF_FRAME_COUNT];
    uint64_t duplicate, stale, invalid;
} perf_frame_state;
static inline perf_frame_record *perf_frame_slot(perf_frame_state *s, uint64_t id) {
    perf_frame_record *r = &s->frames[id % PERF_FRAME_COUNT];
    if (r->present && r->id > id) { s->stale++; return NULL; }
    if (!r->present || r->id != id) { memset(r, 0, sizeof *r); r->id = id; r->present = 1; }
    return r;
}
static inline void perf_frame_start_record(perf_frame_state *s, uint64_t id, uint64_t now, uint64_t pace, uint64_t pq) {
    pthread_mutex_lock(&s->mutex);
    perf_frame_record *r = perf_frame_slot(s, id);
    if (r && !r->closed) { if (!r->start || now < r->start) r->start = now; r->pace = pace; r->pq = pq; }
    else if (r) s->duplicate++;
    pthread_mutex_unlock(&s->mutex);
}
static inline void perf_frame_replay_record(perf_frame_state *s, uint64_t id, uint64_t begin, uint64_t end) {
    pthread_mutex_lock(&s->mutex);
    perf_frame_record *r = perf_frame_slot(s, id);
    if (r && !r->closed && begin && end >= begin && UINT64_MAX - r->replay >= end - begin) {
        if (!r->start || begin < r->start) r->start = begin;
        r->replay += end - begin;
    } else if (r) { r->invalid = 1; s->invalid++; }
    pthread_mutex_unlock(&s->mutex);
}
static inline void perf_frame_close_record(perf_frame_state *s, uint64_t id, uint64_t now,
    int64_t work, int64_t cpu, int64_t replay, int64_t gpu) {
    pthread_mutex_lock(&s->mutex);
    perf_frame_record *r = perf_frame_slot(s, id);
    if (r && !r->closed) {
        r->end = now; r->closed = 1; r->work = work;
        r->cpu = cpu;
        if (replay > r->cpu) r->cpu = replay;
        if (r->replay <= INT64_MAX && (int64_t)r->replay > r->cpu) r->cpu = (int64_t)r->replay;
        r->gpu_estimate = gpu > 0 ? gpu : 0;
        if (!r->start || now < r->start || r->cpu < 0 || r->replay > INT64_MAX) r->invalid = 1;
    } else if (r) s->duplicate++;
    pthread_mutex_unlock(&s->mutex);
}
// Open records are read by exact ID; a previous present may clear the legacy start global.
static inline int perf_frame_peek_record(perf_frame_state *s, uint64_t id, perf_frame_record *out) {
    pthread_mutex_lock(&s->mutex);
    const perf_frame_record *r = &s->frames[id % PERF_FRAME_COUNT];
    int ok = r->present && r->id == id;
    if (ok) *out = *r;
    pthread_mutex_unlock(&s->mutex);
    return ok;
}
static inline int perf_frame_copy_record(perf_frame_state *s, uint64_t id, perf_frame_record *out) {
    pthread_mutex_lock(&s->mutex);
    const perf_frame_record *r = &s->frames[id % PERF_FRAME_COUNT];
    int ok = r->present && r->id == id && r->closed;
    if (ok) *out = *r;
    pthread_mutex_unlock(&s->mutex);
    return ok;
}
static inline int perf_frame_mark_reported(perf_frame_state *s, uint64_t id) {
    pthread_mutex_lock(&s->mutex);
    perf_frame_record *r = &s->frames[id % PERF_FRAME_COUNT];
    int ok = r->present && r->id == id && r->closed && !r->reported;
    if (ok) r->reported = 1;
    else s->duplicate++;
    pthread_mutex_unlock(&s->mutex);
    return ok;
}
// CPU and GPU may overlap. Endpoints, never the sum of their durations, define total.
static inline int perf_frame_values(const perf_frame_record *r, uint64_t gpu_begin, uint64_t gpu_end,
    uint64_t gpu_busy, int64_t *start, int64_t *total) {
    if (r->invalid || !r->start || !r->closed || !gpu_begin || gpu_end < gpu_begin ||
        gpu_begin < r->start || r->end < r->start || r->start > INT64_MAX || gpu_busy > INT64_MAX) return 0;
    uint64_t end = gpu_end > r->end ? gpu_end : r->end;
    uint64_t span = end - r->start;
    if (!span || span >= 250000000 || gpu_busy > span || r->cpu < 0 || (uint64_t)r->cpu > span) return 0;
    *start = (int64_t)r->start; *total = (int64_t)span; return 1;
}
#endif


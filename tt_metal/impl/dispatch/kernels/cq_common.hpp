// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/ring_buffer.h"
#include "cq_helpers.hpp"

FORCE_INLINE
uint32_t round_up_pow2(uint32_t v, uint32_t pow2_size) {
    return (v + (pow2_size - 1)) & ~(pow2_size - 1);
}

FORCE_INLINE
uint32_t div_up(uint32_t n, uint32_t d) {
    return (n + d - 1) / d;
}

FORCE_INLINE
uint32_t wrap_ge(uint32_t a, uint32_t b) {

    // Careful below: have to take the signed diff for 2s complement to handle the wrap
    // Below relies on taking the diff first then the compare to move the wrap
    // to 2^31 away
    int32_t diff = a - b;
    return diff >= 0;
}

FORCE_INLINE
uint32_t wrap_gt(uint32_t a, uint32_t b) {

    // Careful below: have to take the signed diff for 2s complement to handle the wrap
    // Below relies on taking the diff first then the compare to move the wrap
    // to 2^31 away
    int32_t diff = a - b;
    return diff > 0;
}

// The fast CQ noc commands write a subset of the NOC registers for each transaction
// leveraging the fact that many transactions re-use certain values (eg, length)
// Since there are a variety of dispatch paradigms, which values get reused
// depend on the fn
// Making template fns w/ a long list of booleans makes understanding what
// is/not sent tedious
// This is an attempt to pack that data in a way thats ~easy to visually parse
// S/s: send, do not send src address
// N/n: send, do not send noc address
// D/d: send, do not send dst address
// L/l: send, do not send length
constexpr uint32_t CQ_NOC_FLAG_SRC = 0x01;
constexpr uint32_t CQ_NOC_FLAG_NOC = 0x02;
constexpr uint32_t CQ_NOC_FLAG_DST = 0x04;
constexpr uint32_t CQ_NOC_FLAG_LEN = 0x08;
enum CQNocFlags {
    CQ_NOC_sndl = 0,
    CQ_NOC_sndL =                                                       CQ_NOC_FLAG_LEN,
    CQ_NOC_snDl =                                     CQ_NOC_FLAG_DST,
    CQ_NOC_snDL =                                     CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_sNdl =                   CQ_NOC_FLAG_NOC,
    CQ_NOC_sNdL =                   CQ_NOC_FLAG_NOC                   | CQ_NOC_FLAG_LEN,
    CQ_NOC_sNDl =                   CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_sNDL =                   CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_Sndl = CQ_NOC_FLAG_SRC,
    CQ_NOC_SndL = CQ_NOC_FLAG_SRC                                     | CQ_NOC_FLAG_LEN,
    CQ_NOC_SnDl = CQ_NOC_FLAG_SRC                   | CQ_NOC_FLAG_DST,
    CQ_NOC_SnDL = CQ_NOC_FLAG_SRC                   | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
    CQ_NOC_SNdl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC,
    CQ_NOC_SNdL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC                   | CQ_NOC_FLAG_LEN,
    CQ_NOC_SNDl = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST,
    CQ_NOC_SNDL = CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_NOC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN,
};
enum CQNocWait {
    CQ_NOC_wait = 0,
    CQ_NOC_WAIT = 1,
};
enum CQNocSend {
    CQ_NOC_send = 0,
    CQ_NOC_SEND = 1,
};

template<enum CQNocFlags flags, enum CQNocWait wait = CQ_NOC_WAIT, enum CQNocSend send = CQ_NOC_SEND>
FORCE_INLINE
void cq_noc_async_write_with_state(uint32_t src_addr, uint64_t dst_addr, uint32_t size = 0, uint32_t ndests = 1) {

    if constexpr (wait) {
        DEBUG_STATUS("NSSW");
        while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_CMD_BUF));
        DEBUG_STATUS("NSSD");
    }

    if constexpr (flags & CQ_NOC_FLAG_SRC) {
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_TARG_ADDR_LO, src_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_DST) {
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_LO, (uint32_t)dst_addr);
    }
    if constexpr (flags & CQ_NOC_FLAG_NOC) {
#ifdef ARCH_BLACKHOLE
        // Handles writing to PCIe
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_MID, (uint32_t)(dst_addr >> 32) & 0x1000000F);
#endif
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    }
    if constexpr (flags & CQ_NOC_FLAG_LEN) {
        ASSERT(size <= NOC_MAX_BURST_SIZE);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_AT_LEN_BE, size);
     }
    if constexpr (send) {
        DEBUG_SANITIZE_NOC_WRITE_TRANSACTION_FROM_STATE(noc_index);
        NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    }
}

// More generic version of cq_noc_async_write_with_state: Allows writing an abitrary amount of data, when the NOC config (dst_noc,
// VC..) have been specified.
template<bool write_last_packet = true>
FORCE_INLINE
uint32_t cq_noc_async_write_with_state_any_len(uint32_t src_addr, uint64_t dst_addr, uint32_t size = 0, uint32_t ndests = 1) {
    if (size > NOC_MAX_BURST_SIZE) {
        cq_noc_async_write_with_state<CQ_NOC_SnDL>(src_addr, dst_addr, NOC_MAX_BURST_SIZE, ndests);
        src_addr += NOC_MAX_BURST_SIZE;
        dst_addr += NOC_MAX_BURST_SIZE;
        size -= NOC_MAX_BURST_SIZE;
        while(size > NOC_MAX_BURST_SIZE) {
            cq_noc_async_write_with_state<CQ_NOC_SnDl>(src_addr, dst_addr, NOC_MAX_BURST_SIZE, ndests);
            src_addr += NOC_MAX_BURST_SIZE;
            dst_addr += NOC_MAX_BURST_SIZE;
            size -= NOC_MAX_BURST_SIZE;
        }
    }
    if constexpr (write_last_packet) {
        cq_noc_async_write_with_state<CQ_NOC_SnDL>(src_addr, dst_addr, size, ndests);
        return 0;
    } else {
        return size;
    }
}

template<enum CQNocFlags flags, bool mcast = false, bool linked = false>
FORCE_INLINE
void cq_noc_async_write_init_state(uint32_t src_addr, uint64_t dst_addr, uint32_t size = 0) {

    DEBUG_STATUS("NSIW");
    uint32_t heartbeat = 0;
    while (!noc_cmd_buf_ready(noc_index, NCRISC_WR_CMD_BUF)) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    }
    DEBUG_STATUS("NSID");

    constexpr bool multicast_path_reserve = true;
    constexpr bool posted = false;
    constexpr uint32_t vc = mcast ? NOC_DISPATCH_MULTICAST_WRITE_VC : NOC_UNICAST_WRITE_VC;

    constexpr uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR |
        NOC_CMD_VC_STATIC  |
        NOC_CMD_STATIC_VC(vc) |
        (linked ? NOC_CMD_VC_LINKED : 0x0) |
        (mcast ? ((multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) | NOC_CMD_BRCST_PACKET) : 0x0) |
        (posted ? 0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc_index, NCRISC_WR_CMD_BUF, NOC_CTRL, noc_cmd_field);

    cq_noc_async_write_with_state<flags, CQ_NOC_wait, CQ_NOC_send>(src_addr, dst_addr, size);
}

template<uint32_t sem_id>
FORCE_INLINE
void cb_wait_all_pages(uint32_t n) {
    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    DEBUG_STATUS("TAPW");
    while ((*sem_addr) != n);
    DEBUG_STATUS("TAPD");
}

template<uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE
void cb_acquire_pages(uint32_t n) {

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    // Ensure last sem_inc has landed
    noc_async_atomic_barrier();

    DEBUG_STATUS("DAPW");
    // Use a wrapping compare here to compare distance
    // Required for trace which steals downstream credits and may make the value negative
    uint32_t heartbeat = 0;
    while (wrap_gt(n, *sem_addr)) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
    }
    DEBUG_STATUS("DAPD");
    noc_semaphore_inc(get_noc_addr_helper(noc_xy, (uint32_t)sem_addr), -n);
}

template<uint8_t noc_idx, uint32_t noc_xy, uint32_t sem_id>
FORCE_INLINE
void cb_release_pages(uint32_t n) {
    noc_semaphore_inc(get_noc_addr_helper(noc_xy, get_semaphore(sem_id)), n, noc_idx);
}

template<uint32_t noc_xy,
         uint32_t sem_id,
         uint32_t cb_log_page_size>
FORCE_INLINE
uint32_t cb_acquire_pages(uint32_t cb_fence,
                          uint32_t block_next_start_addr[],
                          uint32_t rd_block_idx,
                          uint32_t& local_count) {

    volatile tt_l1_ptr uint32_t* sem_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));

    static uint32_t upstream_count = 0;

    if (local_count == upstream_count) {
        DEBUG_STATUS("UAPW");
        uint32_t heartbeat = 0;
        while ((upstream_count = *sem_addr) == local_count) {
            IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat, 0);
        }
        DEBUG_STATUS("UAPD");
    }

    // Set a fence to limit how much is processed at once
    uint32_t limit = (block_next_start_addr[rd_block_idx] - cb_fence) >> cb_log_page_size;
    uint32_t available = upstream_count - local_count;
    uint32_t usable = (available > limit) ? limit : available;

    local_count += usable;

    return usable;
}

template<uint8_t noc_idx,
         uint32_t noc_xy,
         uint32_t sem_id,
         uint32_t cb_blocks,
         uint32_t cb_pages_per_block>
FORCE_INLINE
void cb_block_release_pages(uint32_t block_noc_writes_to_clear[],
                            uint32_t& wr_block_idx) {

    uint32_t sem_addr = get_semaphore(sem_id);

    uint32_t noc_progress = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
    if (wrap_ge(noc_progress, block_noc_writes_to_clear[wr_block_idx])) {
        noc_semaphore_inc(get_noc_addr_helper(noc_xy, sem_addr), cb_pages_per_block, noc_idx);
        wr_block_idx++;
        wr_block_idx &= (cb_blocks - 1);

        // if >cb_pages_per_block are in flight away from this core
        // then we can fall behind by a block and never catch up
        // checking twice ensures we "gain" on the front if possible
        if (wrap_ge(noc_progress, block_noc_writes_to_clear[wr_block_idx])) {
            noc_semaphore_inc(get_noc_addr_helper(noc_xy, sem_addr), cb_pages_per_block, noc_idx);
            wr_block_idx++;
            wr_block_idx &= (cb_blocks - 1);
        }
    }
}

template<uint32_t cb_blocks>
FORCE_INLINE
void move_rd_to_next_block(uint32_t block_noc_writes_to_clear[],
                           uint32_t& rd_block_idx) {

    // This is subtle: in the free-running case, we don't want to clear the current block
    // if the noc catches up so we artificially inflate the clear value by 1 when we start
    // a block and adjust it down by 1 here as we complete a block
    uint32_t write_count = block_noc_writes_to_clear[rd_block_idx];
    block_noc_writes_to_clear[rd_block_idx] = write_count - 1;

    static_assert((cb_blocks & (cb_blocks - 1)) == 0);
    rd_block_idx++;
    rd_block_idx &= cb_blocks - 1;

    block_noc_writes_to_clear[rd_block_idx] = write_count; // this is plus 1
}

template<uint32_t cb_base,
         uint32_t cb_blocks,
         uint32_t cb_log_page_size,
         uint32_t noc_xy,
         uint32_t cb_sem>
FORCE_INLINE
uint32_t get_cb_page(uint32_t& cmd_ptr,
                     uint32_t& cb_fence,
                     uint32_t block_noc_writes_to_clear[],
                     uint32_t block_next_start_addr[],
                     uint32_t& rd_block_idx,
                     uint32_t& local_count) {

    // Strided past the data that has arrived, get the next page
    if (cb_fence == block_next_start_addr[rd_block_idx]) {
        if (rd_block_idx == cb_blocks - 1) {
            cmd_ptr = cb_base;
            cb_fence = cb_base;
        }
        move_rd_to_next_block<cb_blocks>(block_noc_writes_to_clear,
                                         rd_block_idx);
    }

    // Wait for dispatcher to supply a page
    uint32_t n_pages = cb_acquire_pages<noc_xy,
                                        cb_sem,
                                        cb_log_page_size>(cb_fence,
                                                          block_next_start_addr,
                                                          rd_block_idx,
                                                          local_count);
    cb_fence += n_pages << cb_log_page_size;

    return n_pages;
}

constexpr uint32_t l1_to_local_cache_copy_chunk = 6;

// NOTE: CAREFUL USING THIS FUNCTION
// It is call "careful_copy" because you need to be careful...
// It copies beyond count by up to 5 elements make sure src and dst addresses are safe
template<uint32_t l1_to_local_cache_copy_chunk, uint32_t l1_cache_elements_rounded>
FORCE_INLINE
void careful_copy_from_l1_to_local_cache(volatile uint32_t tt_l1_ptr *l1_ptr, uint32_t count, uint32_t * l1_cache) {
    uint32_t n = 0;
    ASSERT(l1_to_local_cache_copy_chunk == 6);
    ASSERT(count <= l1_cache_elements_rounded);
    while (n < count) {
        uint32_t v0 = l1_ptr[n + 0];
        uint32_t v1 = l1_ptr[n + 1];
        uint32_t v2 = l1_ptr[n + 2];
        uint32_t v3 = l1_ptr[n + 3];
        uint32_t v4 = l1_ptr[n + 4];
        uint32_t v5 = l1_ptr[n + 5];
        l1_cache[n + 0] = v0;
        l1_cache[n + 1] = v1;
        l1_cache[n + 2] = v2;
        l1_cache[n + 3] = v3;
        l1_cache[n + 4] = v4;
        l1_cache[n + 5] = v5;
        n += 6;
    }
}

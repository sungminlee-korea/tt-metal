#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {

  // TODO. get runtime arguments.
  uint32_t arg = 0;
  const auto cb0_id = get_arg_val<uint32_t>(arg++);
  const auto cb1_id = get_arg_val<uint32_t>(arg++);
  const auto num_tiles = get_arg_val<uint32_t>(arg++);

  constexpr auto dst0 = 0;
  constexpr auto first = 0;
  constexpr auto onetile = 1;

  unary_op_init_common(cb0_id, cb1_id);

  for (uint32_t i = 0; i < num_tiles; i++) {

    tile_regs_acquire();
    cb_wait_front(cb0_id, onetile);
    copy_tile_init();
    copy_tile(cb0_id, first, dst0);
    cb_pop_front(cb0_id, onetile);

    relu_tile_init();
    relu_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb1_id, onetile);
    pack_tile(dst0, cb1_id, first);
    cb_push_back(cb1_id, onetile);
    tile_regs_release();
  }

  UNPACK(DPRINT << "UNPACK END" << ENDL());
  MATH(DPRINT << "MATH END" << ENDL());
  PACK(DPRINT << "PACK END" << ENDL());
}
} // namespace NAMESPACE

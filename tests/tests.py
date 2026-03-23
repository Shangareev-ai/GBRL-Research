import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from numpy.testing import assert_allclose
from environment import AssetEpisodeLog, AssetEnvState

def test_asset_episode_log():
    print("=== AssetEpisodeLog tests ===\n")

    # ── 1. init ───────────────────────────────────────────────────
    log = AssetEpisodeLog(initial_size=4)
    assert log.size == 0
    assert log.capacity == 4
    print(f"[PASS] init: size={log.size}, capacity={log.capacity}")

    # ── 2. push_signal ────────────────────────────────────────────
    # activate (pass signal)
    log.push_signal(np.int64(1_000_000_000), np.int8(0), np.float32(0), np.float32(0), np.float32(100_000), np.float32(0))
    assert log.size == 1
    assert log.signal_types[0] == 0
    assert log.cash[0] == np.float32(100_000)
    assert log.timestamps[0] == 1_000_000_000
    print(f"[PASS] push_signal: size after activate={log.size}, ts={log.timestamps[0]}")

    # open_long
    log.push_signal(np.int64(2_000_000_000), np.int8(1), np.float32(2.0), np.float32(0.5), np.float32(0), np.float32(50_000))
    assert log.size == 2
    assert log.signal_types[1] == 1
    assert log.leverage[1] == np.float32(2.0)
    print(f"[PASS] push_signal: size after open_long={log.size}")

    # close_long
    log.push_signal(np.int64(3_000_000_000), np.int8(4), np.float32(0), np.float32(0), np.float32(110_000), np.float32(0))
    assert log.size == 3
    assert log.signal_types[2] == 4
    print(f"[PASS] push_signal: size after close_long={log.size}")

    # ── 3. resize (push beyond initial_size=4) ───────────────────
    log.push_signal(np.int64(4_000_000_000), np.int8(-1), np.float32(1.0), np.float32(-0.3), np.float32(0), np.float32(-110_000))
    assert log.size == 4 and log.capacity == 4          # exactly full, no resize yet

    log.push_signal(np.int64(5_000_000_000), np.int8(-4), np.float32(0), np.float32(0), np.float32(95_000), np.float32(0))
    assert log.size == 5 and log.capacity == 8          # resize triggered: 4 → 8
    # verify data survived resize
    assert log.signal_types[3] == -1
    assert log.signal_types[4] == -4
    assert log.cash[4] == np.float32(95_000)
    assert log.timestamps[4] == 5_000_000_000
    print(f"[PASS] resize: capacity doubled to {log.capacity}, size={log.size}, data intact")

    # ── 4. get_logged_signals ─────────────────────────────────────
    timestamps, stypes, levs, psizes, cash, debt = log.get_logged_signals()
    assert len(timestamps) == 5
    assert len(stypes)     == 5
    assert list(stypes)    == [0, 1, 4, -1, -4]
    assert timestamps[0] == 1_000_000_000
    assert timestamps[4] == 5_000_000_000
    print(f"[PASS] get_logged_signals: {len(timestamps)} entries, signal_types={list(stypes)}")

    # ── 5. frozen_copy ────────────────────────────────────────────
    frozen = log.frozen_copy()
    assert frozen.size     == log.size      # same entries
    assert frozen.capacity == log.size      # trimmed — no wasted allocation
    assert frozen.capacity < log.capacity   # smaller than original (5 vs 8)

    f_timestamps, f_stypes, f_levs, f_psizes, f_cash, f_debt = frozen.get_logged_signals()
    assert list(f_stypes)     == list(stypes)
    assert list(f_cash)       == list(cash)
    assert list(f_timestamps) == list(timestamps)
    print(f"[PASS] frozen_copy: capacity={frozen.capacity} (trimmed from {log.capacity}), data identical")

    # mutating original does not affect frozen
    log.push_signal(np.int64(6_000_000_000), np.int8(0), np.float32(0), np.float32(0), np.float32(99_000), np.float32(0))
    assert frozen.size == 5 and log.size == 6
    print(f"[PASS] frozen_copy is independent: frozen.size={frozen.size}, log.size={log.size}")

    print("\n[PASS] All AssetEpisodeLog tests passed.")

def test_asset_env_state():
    print("=== AssetEnvState tests ===\n")

    COMM = 1e-3

    # ── 1. open_long (new position) ────────────────────────────────
    s = AssetEnvState(initial_cash=10_000.0, commission_rate=COMM,
                      max_leverage=2.0, allow_shorts=0)
    s.reset(np.float32(100.0), np.int64(1_000_000_000))
    s.open_long(np.float32(100.0), np.int64(2_000_000_000), np.float32(2.0))

    exp_coin = np.float32(20_000.0 * (1 - COMM) / 100.0)   # 199.8
    exp_debt = np.float32(10_000.0)
    exp_liq  = float(exp_debt) / float(exp_coin)             # ≈ 50.0501

    assert s.position == 1
    assert s.cash == np.float32(0.0)
    assert_allclose(s.coin, exp_coin, rtol=1e-5)
    assert_allclose(s.debt, exp_debt, rtol=1e-5)
    assert_allclose(s.liquidation_price, exp_liq, rtol=1e-5)
    print(f"[PASS] open_long:  coin={s.coin:.4f}, debt={s.debt:.1f}, liq_price={s.liquidation_price:.4f}")

    # ── 2. close_long ─────────────────────────────────────────────
    s.close_long(np.float32(110.0), np.int64(3_000_000_000))

    exp_proceeds = float(exp_coin) * 110.0 * (1 - COMM)
    exp_cash     = exp_proceeds - float(exp_debt)

    assert s.position == 0
    assert s.coin == np.float32(0.0) and s.debt == np.float32(0.0)
    assert_allclose(s.cash, exp_cash, rtol=1e-4)
    print(f"[PASS] close_long at 110: cash={s.cash:.4f}  (expected {exp_cash:.4f})")

    # ── 3. open_short (new position) ──────────────────────────────
    s2 = AssetEnvState(initial_cash=10_000.0, commission_rate=COMM,
                       max_leverage=2.0, allow_shorts=1)
    s2.reset(np.float32(100.0), np.int64(1_000_000_000))
    s2.open_short(np.float32(100.0), np.int64(2_000_000_000), np.float32(2.0))

    exp_borrowed      = np.float32(10_000.0 * 2.0 / 100.0)
    exp_cash_obtained = float(exp_borrowed) * 100.0 * (1 - COMM)
    exp_locked        = exp_cash_obtained + 10_000.0
    exp_liq_short     = exp_locked / float(exp_borrowed)

    assert s2.position == -1 and s2.cash == np.float32(0.0)
    assert_allclose(s2.coin, -exp_borrowed, rtol=1e-5)
    assert_allclose(s2.debt, -exp_locked,   rtol=1e-5)
    assert_allclose(s2.liquidation_price, exp_liq_short, rtol=1e-5)
    print(f"[PASS] open_short: coin={s2.coin:.4f}, debt={s2.debt:.1f}, liq_price={s2.liquidation_price:.4f}")

    # ── 4. close_short ────────────────────────────────────────────
    s2.close_short(np.float32(90.0), np.int64(3_000_000_000))

    exp_cash_used = float(exp_borrowed) * 90.0 * (1 + COMM)
    exp_cash_end  = exp_locked - exp_cash_used

    assert s2.position == 0 and s2.coin == np.float32(0.0) and s2.debt == np.float32(0.0)
    assert_allclose(s2.cash, exp_cash_end, rtol=1e-4)
    print(f"[PASS] close_short at 90: cash={s2.cash:.4f}  (expected {exp_cash_end:.4f})")

    # ── 5. check_liquidation (long) ───────────────────────────────
    s3 = AssetEnvState(initial_cash=10_000.0, commission_rate=COMM,
                       max_leverage=2.0, allow_shorts=0)
    s3.reset(np.float32(100.0), np.int64(1_000_000_000))
    s3.open_long(np.float32(100.0), np.int64(2_000_000_000), np.float32(2.0))
    liq_p = s3.liquidation_price
    s3.check_liquidation(np.float32(liq_p - 1.0), np.int64(3_000_000_000))
    assert s3.position == 0, f"long not liquidated (position={s3.position})"
    print(f"[PASS] check_liquidation (long):  fired at {liq_p-1.0:.2f} < liq={liq_p:.4f}")

    # ── 6. check_liquidation (short) ──────────────────────────────
    s4 = AssetEnvState(initial_cash=10_000.0, commission_rate=COMM,
                       max_leverage=2.0, allow_shorts=1)
    s4.reset(np.float32(100.0), np.int64(1_000_000_000))
    s4.open_short(np.float32(100.0), np.int64(2_000_000_000), np.float32(2.0))
    liq_p4 = s4.liquidation_price
    s4.check_liquidation(np.float32(liq_p4 + 1.0), np.int64(3_000_000_000))
    assert s4.position == 0, f"short not liquidated (position={s4.position})"
    print(f"[PASS] check_liquidation (short): fired at {liq_p4+1.0:.2f} > liq={liq_p4:.4f}")

    # ── 7. step() — (state, reward, terminated) ───────────────────
    # action=0.5 → target_lev = 0.5×2 = 1.0 (long, 1× = no borrowing)
    #   coin = 10000×(1−comm)/100 = 99.9,  equity = 9990
    #   reward ≈ log(9990/10000) ≈ −0.001  (commission drag on open)
    s5 = AssetEnvState(initial_cash=10_000.0, commission_rate=COMM,
                       max_leverage=2.0, allow_shorts=1)
    s5.reset(np.float32(100.0), np.int64(0))

    state, r_open, done_open = s5.step(
        np.float32(0.5), np.float32(100.0), np.int8(0), np.int64(1_000_000_000))
    assert s5.position == 1 and s5.leverage == np.float32(1.0)
    assert state['position'] == np.float32(1.0)
    assert done_open == int8(0)
    assert_allclose(r_open, np.log(9990.0 / 10_000.0), atol=1e-4)
    print(f"[PASS] step open_long:  reward={r_open:.6f}  state={dict(state)}  done={done_open}")

    # hold: price +1% → reward ≈ log(1.01)
    state, r_hold, done_hold = s5.step(
        np.float32(0.5), np.float32(101.0), np.int8(0), np.int64(2_000_000_000))
    assert r_hold > 0.0 and done_hold == int8(0)
    assert state['position'] == np.float32(1.0)
    assert_allclose(r_hold, np.log(101.0 / 100.0), atol=2e-3)
    print(f"[PASS] step hold +1%:   reward={r_hold:.6f}  (expected {np.log(1.01):.6f})")

    # flatten: action=0 → close_long; negative reward (commission)
    state, r_flat, done_flat = s5.step(
        np.float32(0.0), np.float32(101.0), np.int8(0), np.int64(3_000_000_000))
    assert s5.position == 0 and r_flat < 0.0 and done_flat == int8(0)
    assert state['position'] == np.float32(0.0)
    print(f"[PASS] step flatten:    reward={r_flat:.6f}  state={dict(state)}  done={done_flat}")

    # turb=1 forces flat regardless of action
    s6 = AssetEnvState(initial_cash=10_000.0, commission_rate=COMM,
                       max_leverage=2.0, allow_shorts=1)
    s6.reset(np.float32(100.0), np.int64(0))
    s6.open_long(np.float32(100.0), np.int64(1_000_000_000), np.float32(2.0))
    state6, _, done6 = s6.step(
        np.float32(1.0), np.float32(100.0), np.int8(1), np.int64(2_000_000_000))
    assert s6.position == 0 and done6 == int8(0)
    assert state6['position'] == np.float32(0.0)
    print(f"[PASS] step turb=1:     position forced flat, state={dict(state6)}")

    # terminated=1 when total_asset ≤ 0 (simulate bankruptcy manually)
    s7 = AssetEnvState(initial_cash=10_000.0, commission_rate=COMM,
                       max_leverage=2.0, allow_shorts=0)
    s7.reset(np.float32(100.0), np.int64(0))
    s7.cash = np.float32(-1.0)   # force bankrupt state
    state7, r7, done7 = s7.step(
        np.float32(0.0), np.float32(100.0), np.int8(0), np.int64(1_000_000_000))
    assert done7 == int8(1), f"expected terminated=1, got {done7}"
    print(f"[PASS] terminated=1 when total_asset≤0: done={done7}")

    # ── 8. open_long — increase leverage (else branch) ────────────
    # Open 1× at 100 → entry=100, coin=99.9, debt=0, collateral=10_000
    # Increase to 2× at 110 → weighted avg entry, borrow more, add coin
    s8 = AssetEnvState(initial_cash=10_000.0, commission_rate=COMM,
                       max_leverage=3.0, allow_shorts=0)
    s8.reset(np.float32(100.0), np.int64(0))
    s8.open_long(np.float32(100.0), np.int64(1_000_000_000), np.float32(1.0))

    exp_coin_1x = float(np.float32(10_000.0 * (1 - COMM) / 100.0))   # 99.9
    assert_allclose(s8.coin, exp_coin_1x, rtol=1e-5)
    assert s8.debt == np.float32(0.0) and s8.collateral == np.float32(10_000.0)

    s8.open_long(np.float32(110.0), np.int64(2_000_000_000), np.float32(2.0))

    # entry = (old_lev/lev)*entry_old + ((lev-old_lev)/lev)*price
    #       = (1/2)*100 + (1/2)*110 = 105
    exp_entry_inc  = (1.0 / 2.0) * 100.0 + (1.0 / 2.0) * 110.0   # 105.0
    exp_borrowed   = 10_000.0 * (2.0 - 1.0)                        # collateral*(lev-old)=10_000
    exp_dcoin      = exp_borrowed * (1 - COMM) / 110.0
    exp_coin_inc   = exp_coin_1x + exp_dcoin
    exp_debt_inc   = 0.0 + exp_borrowed                             # 10_000
    exp_liq_inc    = exp_debt_inc / exp_coin_inc

    assert s8.position == 1 and s8.leverage == np.float32(2.0) and s8.cash == np.float32(0.0)
    assert_allclose(s8.entry_price,       exp_entry_inc, rtol=1e-5)
    assert_allclose(s8.coin,              exp_coin_inc,  rtol=1e-5)
    assert_allclose(s8.debt,              exp_debt_inc,  rtol=1e-5)
    assert_allclose(s8.liquidation_price, exp_liq_inc,   rtol=1e-5)
    assert s8.log.signal_types[s8.log.size - 1] == np.int8(2)   # open_long+ signal
    print(f"[PASS] open_long increase: entry={s8.entry_price:.2f} (exp {exp_entry_inc:.2f}), "
          f"coin={s8.coin:.4f}, debt={s8.debt:.1f}, liq={s8.liquidation_price:.4f}")

    # ── 9. open_short — increase leverage (else branch) ───────────
    # Open 1× at 100 → coin=-100, debt=-19_990, collateral=10_000
    # Increase to 2× at 90  → weighted avg entry, borrow more coin,
    #   only new proceeds (cash_obtained) added to locked, not collateral again
    s9 = AssetEnvState(initial_cash=10_000.0, commission_rate=COMM,
                       max_leverage=3.0, allow_shorts=1)
    s9.reset(np.float32(100.0), np.int64(0))
    s9.open_short(np.float32(100.0), np.int64(1_000_000_000), np.float32(1.0))

    exp_borrow_1xs = 10_000.0 * 1.0 / 100.0                       # 100.0
    exp_locked_1xs = exp_borrow_1xs * 100.0 * (1 - COMM) + 10_000.0  # 19_990
    assert_allclose(s9.coin, -exp_borrow_1xs, rtol=1e-5)
    assert_allclose(s9.debt, -exp_locked_1xs, rtol=1e-5)
    assert s9.collateral == np.float32(10_000.0)

    s9.open_short(np.float32(90.0), np.int64(2_000_000_000), np.float32(2.0))

    # entry = (1/2)*100 + (1/2)*90 = 95
    exp_entry_incs  = (1.0 / 2.0) * 100.0 + (1.0 / 2.0) * 90.0   # 95.0
    exp_dcoin_s     = 10_000.0 * (2.0 - 1.0) / 90.0               # collateral*(lev-old)/price
    exp_cash_obt_s  = exp_dcoin_s * 90.0 * (1 - COMM)             # only new proceeds locked
    exp_coin_incs   = -(exp_borrow_1xs + exp_dcoin_s)
    exp_debt_incs   = -(exp_locked_1xs + exp_cash_obt_s)
    exp_liq_incs    = (-exp_debt_incs) / (-exp_coin_incs)

    assert s9.position == -1 and s9.leverage == np.float32(2.0) and s9.cash == np.float32(0.0)
    assert_allclose(s9.entry_price,       exp_entry_incs, rtol=1e-5)
    assert_allclose(s9.coin,              exp_coin_incs,  rtol=1e-5)
    assert_allclose(s9.debt,              exp_debt_incs,  rtol=1e-5)
    assert_allclose(s9.liquidation_price, exp_liq_incs,   rtol=1e-5)
    assert s9.log.signal_types[s9.log.size - 1] == np.int8(-2)  # open_short+ signal
    print(f"[PASS] open_short increase: entry={s9.entry_price:.2f} (exp {exp_entry_incs:.2f}), "
          f"coin={s9.coin:.4f}, debt={s9.debt:.1f}, liq={s9.liquidation_price:.4f}")

    print("\n[PASS] All AssetEnvState tests passed.")

test_asset_env_state()



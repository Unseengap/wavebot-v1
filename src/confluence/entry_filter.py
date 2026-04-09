"""Entry filter: all conditions must pass before a trade is submitted."""


def check_entry_conditions(
    instrument: str,
    direction: str,
    confluence_score: float,
    wave_scores: dict,
    gate: str,
    open_trades: list,
    current_spread_pips: float,
    max_spread_pips: float,
    signal_frame_ws,
    rr_ratio: float,
    daily_drawdown_pct: float,
    session: str,
    config: dict,
) -> tuple:
    """
    Returns (can_enter: bool, reasons: list[str])
    All checks must pass.
    """
    fails = []
    threshold = config.get("min_confluence_score", 0.65)
    min_frames = config.get("min_frames_aligned", 2)
    max_trades = config.get("max_open_trades", 3)
    max_maturity = config.get("max_entry_maturity", 0.75)
    min_rr = config.get("min_rr_ratio", 2.0)
    max_dd = config.get("max_daily_drawdown", 0.02)

    # 1. Confluence threshold
    if direction == "LONG" and confluence_score < threshold:
        fails.append(f"Score {confluence_score:.3f} < {threshold}")
    elif direction == "SHORT" and confluence_score > -threshold:
        fails.append(f"Score {confluence_score:.3f} > -{threshold}")

    # 2. Directional gate
    if gate == "NEUTRAL":
        fails.append("Gate is NEUTRAL — no entries")
    elif direction == "LONG" and gate == "BEARISH":
        if abs(confluence_score) < 0.85:
            fails.append("LONG against BEARISH gate (need score > 0.85)")
    elif direction == "SHORT" and gate == "BULLISH":
        if abs(confluence_score) < 0.85:
            fails.append("SHORT against BULLISH gate (need score > 0.85)")

    # 3. Minimum frames aligned
    aligned = sum(
        1 for ws in wave_scores.values()
        if ws is not None and ws.granularity in ("M1","M5","M15","H1","H4") and (
            (direction == "LONG" and ws.direction > 0) or
            (direction == "SHORT" and ws.direction < 0)
        )
    )
    if aligned < min_frames:
        fails.append(f"Only {aligned} frames aligned (need {min_frames}+)")

    # 4. No duplicate position
    for t in open_trades:
        if t["instrument"] == instrument:
            fails.append(f"Already in position on {instrument}")
            break

    # 5. Max open trades
    if len(open_trades) >= max_trades:
        fails.append(f"Max trades ({max_trades}) reached")

    # 6. Spread
    if current_spread_pips > max_spread_pips:
        fails.append(f"Spread {current_spread_pips:.1f} > max {max_spread_pips:.1f}")

    # 7. Signal wave maturity
    if signal_frame_ws and signal_frame_ws.maturity > max_maturity:
        fails.append(f"Maturity {signal_frame_ws.maturity:.2f} > {max_maturity}")

    # 8. Minimum R:R
    if rr_ratio < min_rr:
        fails.append(f"R:R {rr_ratio:.2f} < {min_rr}")

    # 9. Daily drawdown
    if daily_drawdown_pct >= max_dd:
        fails.append("Daily drawdown limit reached")

    # 10. Dead zone
    if session == "Dead_Zone":
        fails.append("Dead Zone — no entries")

    return (len(fails) == 0, fails)

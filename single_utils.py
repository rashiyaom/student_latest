# =============================================================================
# single_utils.py
# ClassPulse AI — Single-Student HUD Drawing
# All drawing for the single_student_main.py live view window
# =============================================================================

import cv2
import numpy as np

FONT     = cv2.FONT_HERSHEY_SIMPLEX
C_GREEN  = (0,   210, 0)
C_RED    = (0,   0,   220)
C_ORANGE = (0,   140, 255)
C_YELLOW = (0,   220, 220)
C_CYAN   = (210, 210, 0)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,   0,   0)
C_PURPLE = (200, 0,   200)
C_PHONE  = (0,   0,   255)

PERSON_CLASS = 0
PHONE_CLASS  = 67


def _label(frame, text, x, y, fg=C_WHITE, bg=(20,20,20), fs=0.42, th=1):
    (tw, th_), bl = cv2.getTextSize(text, FONT, fs, th)
    p = 3
    cv2.rectangle(frame, (x-p, y-th_-p), (x+tw+p, y+bl+p), bg, cv2.FILLED)
    cv2.putText(frame, text, (x,y), FONT, fs, fg, th, cv2.LINE_AA)


def _semi_rect(frame, x1, y1, x2, y2, color, alpha=0.5):
    ov = frame.copy()
    cv2.rectangle(ov, (x1,y1),(x2,y2), color, cv2.FILLED)
    cv2.addWeighted(ov, alpha, frame, 1-alpha, 0, frame)


def draw_single_student_hud(frame, state, person_box, all_dets, fps):
    """
    Draw the full live HUD onto frame in-place.
    """
    fh, fw = frame.shape[:2]

    # ------------------------------------------------------------------ #
    # 1. Person bounding box
    # ------------------------------------------------------------------ #
    if person_box is not None:
        x1,y1,x2,y2 = person_box
        att   = state.get('attention_label','?')
        score = state.get('attention_score', 0)
        col   = (C_GREEN   if att=='ATTENTIVE' else
                 C_ORANGE  if att=='DISTRACTED' else
                 C_RED)
        cv2.rectangle(frame, (x1,y1),(x2,y2), col, 2)
        _label(frame, "Student [{:.0f}%]".format(score*100),
               x1, y1-5, fg=C_WHITE, bg=col)

    # ------------------------------------------------------------------ #
    # 2. Phone bounding box
    # ------------------------------------------------------------------ #
    phone_dets = [d for d in all_dets if len(d)>=6 and int(d[5])==PHONE_CLASS]
    for d in phone_dets:
        px1,py1,px2,py2 = int(d[0]),int(d[1]),int(d[2]),int(d[3])
        cv2.rectangle(frame, (px1,py1),(px2,py2), C_PHONE, 3)
        _label(frame, "PHONE {:.0f}%".format(d[4]*100),
               px1, py1-5, fg=C_WHITE, bg=C_PHONE, fs=0.45, th=2)

    # PHONE ACTIVE banner
    if state.get('phone_active'):
        dur = state.get('phone_current_secs', 0)
        banner = "  PHONE IN USE — {:.0f}s  ".format(dur)
        (bw, bh), _ = cv2.getTextSize(banner, FONT, 0.65, 2)
        bx = (fw - bw) // 2
        _semi_rect(frame, bx-8, 8, bx+bw+8, 8+bh+14, C_PHONE, 0.75)
        cv2.putText(frame, banner, (bx, 8+bh+4), FONT, 0.65,
                    C_WHITE, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # 3. Head-down banner
    # ------------------------------------------------------------------ #
    if state.get('head_currently_down'):
        streak = state.get('head_down_current_secs', 0)
        total_m= state.get('head_down_total_mins', 0)
        banner = "  HEAD DOWN — {:.0f}s streak | Total: {:.1f}min  ".format(
            streak, total_m)
        (bw, bh), _ = cv2.getTextSize(banner, FONT, 0.55, 1)
        bx = (fw - bw) // 2
        _semi_rect(frame, bx-8, 36, bx+bw+8, 36+bh+12, C_YELLOW, 0.7)
        cv2.putText(frame, banner, (bx, 36+bh+2), FONT, 0.55,
                    C_BLACK, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # 4. YAWN flash
    # ------------------------------------------------------------------ #
    if state.get('yawn_state') == 'YAWNING':
        _semi_rect(frame, 0, 0, fw, fh, C_CYAN, 0.05)
        cv2.putText(frame, "YAWN #{} DETECTED".format(state.get('yawn_count',0)),
                    (8, fh-36), FONT, 0.65, C_CYAN, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # 5. Right-side stats panel
    # ------------------------------------------------------------------ #
    pw = 200
    _semi_rect(frame, fw-pw-6, 0, fw, fh, C_BLACK, 0.55)

    def stat_row(y, label, val, col=C_WHITE):
        cv2.putText(frame, label, (fw-pw+4, y), FONT, 0.38,
                    (150,150,150), 1, cv2.LINE_AA)
        cv2.putText(frame, str(val), (fw-pw+4, y+16), FONT, 0.52,
                    col, 1, cv2.LINE_AA)
        return y + 38

    y = 20
    # Attention score bar
    att_pct = int(state.get('attention_score', 0) * 100)
    att_col = (C_GREEN   if att_pct>=65 else
               C_ORANGE  if att_pct>=35 else C_RED)
    cv2.putText(frame, "ATTENTION", (fw-pw+4, y), FONT, 0.38,
                (150,150,150), 1, cv2.LINE_AA)
    y += 14
    bar_x1 = fw-pw+4; bar_x2 = fw-10
    bar_w  = bar_x2 - bar_x1
    cv2.rectangle(frame, (bar_x1,y),(bar_x2,y+8), (40,40,40), cv2.FILLED)
    cv2.rectangle(frame, (bar_x1,y),(bar_x1+int(bar_w*att_pct/100),y+8),
                  att_col, cv2.FILLED)
    cv2.putText(frame, "{} {}%".format(
        state.get('attention_label','?')[:3], att_pct),
        (bar_x1, y+22), FONT, 0.45, att_col, 1, cv2.LINE_AA)
    y += 32

    cv2.line(frame, (fw-pw+4, y), (fw-10, y), (50,50,50), 1)
    y += 8

    eye_col  = C_RED   if state.get('eye_state')=='DROWSY' else C_GREEN
    head_col = C_YELLOW if state.get('head_state')=='DOWN'  else C_GREEN
    yawn_col = C_CYAN   if state.get('yawn_state')=='YAWNING' else C_GREEN

    y = stat_row(y, "EYE STATE", state.get('eye_state','?'), eye_col)
    y = stat_row(y, "HEAD",      state.get('head_state','?'), head_col)
    y = stat_row(y, "YAWN STATE",state.get('yawn_state','?'), yawn_col)

    cv2.line(frame, (fw-pw+4, y), (fw-10, y), (50,50,50), 1)
    y += 8

    y = stat_row(y, "YAWNS TOTAL",
                 state.get('yawn_count',0), C_CYAN)
    y = stat_row(y, "HEAD-DOWN EVENTS",
                 state.get('head_down_count',0), C_YELLOW)
    y = stat_row(y, "HEAD-DOWN TIME",
                 "{:.1f} min".format(state.get('head_down_total_mins',0)), C_YELLOW)
    y = stat_row(y, "PHONE EVENTS",
                 state.get('phone_events_count',0), C_PHONE)
    y = stat_row(y, "PHONE TIME",
                 "{:.0f}s".format(state.get('phone_total_secs',0)), C_PHONE)

    cv2.line(frame, (fw-pw+4, y), (fw-10, y), (50,50,50), 1)
    y += 8

    # Session attention %
    sess_att = state.get('attention_pct', 0)
    col_sa = C_GREEN if sess_att>=65 else C_ORANGE if sess_att>=35 else C_RED
    cv2.putText(frame, "SESSION ATT%", (fw-pw+4, y), FONT, 0.38,
                (150,150,150), 1, cv2.LINE_AA)
    cv2.putText(frame, "{:.0f}%".format(sess_att),
                (fw-pw+4, y+20), FONT, 0.72, col_sa, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # 6. FPS counter
    # ------------------------------------------------------------------ #
    _label(frame, "FPS:{:.1f}".format(fps), 6, fh-8, fs=0.4)

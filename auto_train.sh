#!/bin/bash
# Idle babysitter for the training loop: starts polyfish-rs/run_training_loop.sh
# only while the desktop is idle inside the allowed window, and halts it as soon
# as the machine is in use again.
#
# Linux/GNOME only by design - it reads idle time from the Mutter idle monitor
# over gdbus and halts by signalling a process group started with setsid.
# No daily report here: run `node run_analysis_now.js` from the repo root, which
# reads polyfish-rs/training_log.csv and needs no curriculum knowledge to stay
# correct.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --resume is mandatory, not decoration: run_training_loop.sh refuses a bare
# launch once model.safetensors and training_log.csv history both exist (#37),
# so without it every restart after the first campaign iteration dies at startup.
TRAIN_SCRIPT="polyfish-rs/run_training_loop.sh"
TRAIN_ARGS=(--resume)
CHECK_INTERVAL=1 # seconds
IDLE_THRESHOLD_SECS=60
HALT_GRACE_SECS=15

TRAIN_DIR="$REPO_ROOT/$(dirname "$TRAIN_SCRIPT")"
TRAIN_BIN="$(basename "$TRAIN_SCRIPT")"

for tool in gdbus setsid; do
    if ! command -v "$tool" &> /dev/null; then
        echo "$tool is required but not found."
        exit 1
    fi
done

if [ ! -x "$TRAIN_DIR/$TRAIN_BIN" ]; then
    echo "$TRAIN_SCRIPT is missing or not executable."
    exit 1
fi

is_allowed_time() {
    local day=$(date +%w) # 0 = Sunday, 1-5 = Mon-Fri, 6 = Saturday
    local hour=$(date +%H)

    # Weekend: All day
    if [[ "$day" == "0" || "$day" == "6" ]]; then
        return 0
    fi

    # Weekday: 20:00 (8 PM) to 08:00 (8 AM)
    if [[ "$hour" -ge 20 || "$hour" -lt 8 ]]; then
        return 0
    fi

    return 1
}

# setsid gives the loop its own session so a halt reaches the whole tree -
# self_play, train.py and the polyfish server the loop starts. A background job
# here is not a process-group leader (no job control in a script), so setsid
# execs in place and $! is the leader of the new group.
start_training() {
    ( cd "$TRAIN_DIR" && exec setsid "./$TRAIN_BIN" "${TRAIN_ARGS[@]}" ) &
    TRAIN_PID=$!
    echo "$(date): Training started (pid/pgid $TRAIN_PID): $TRAIN_SCRIPT ${TRAIN_ARGS[*]}"
}

# TERM the group first so the loop's EXIT trap can run finish-run, kill the
# server it started and clear .training.pid; SIGKILL on the wrapper alone left
# the server holding port 3000 and blocked the next start.
halt_training() {
    echo "$(date): $1 Halting training (pgid $TRAIN_PID)."
    kill -TERM -- -"$TRAIN_PID" 2>/dev/null
    local waited=0
    while kill -0 -- -"$TRAIN_PID" 2>/dev/null && [ "$waited" -lt "$HALT_GRACE_SECS" ]; do
        sleep 1
        waited=$((waited + 1))
    done
    kill -KILL -- -"$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID" 2>/dev/null
    TRAIN_PID=""
}

TRAIN_PID=""
IDLE_SECS=0

echo "Starting auto-train monitor..."
echo "Training command: $TRAIN_SCRIPT ${TRAIN_ARGS[*]} (in $TRAIN_DIR)"
echo "Training allowed: Weekdays 20:00-08:00, Weekends all day."
echo "Requires ${IDLE_THRESHOLD_SECS}s of input inactivity."
echo "Checking every ${CHECK_INTERVAL} seconds..."

while true; do
    # Idle time in milliseconds (works on Wayland GNOME)
    IDLE_MS=$(gdbus call --session --dest org.gnome.Mutter.IdleMonitor --object-path /org/gnome/Mutter/IdleMonitor/Core --method org.gnome.Mutter.IdleMonitor.GetIdletime 2>/dev/null | grep -o "[0-9]\+" | tail -1)

    if [[ -n "$IDLE_MS" ]]; then
        IDLE_SECS=$((IDLE_MS / 1000))
    else
        IDLE_SECS=0
    fi

    if [[ "$IDLE_SECS" -lt "$IDLE_THRESHOLD_SECS" ]]; then
        if [[ -n "$TRAIN_PID" ]]; then
            halt_training "Activity detected."
        fi
    elif is_allowed_time; then
        if [[ -z "$TRAIN_PID" ]]; then
            echo "$(date): Conditions met (idle for ${IDLE_SECS}s)."
            start_training
        elif ! kill -0 "$TRAIN_PID" 2>/dev/null; then
            wait "$TRAIN_PID" 2>/dev/null
            TRAIN_STATUS=$?
            TRAIN_PID=""
            # The loop's exit codes are its safety mechanism: it aborts nonzero on a
            # failed gauge reading, anchor snapshot or link match, and exits 3 on a
            # plateau stop. Restarting through either would undo the decision - a
            # blind retry loop past a fatal abort, or training on past the plateau
            # the gate exists to catch. Only a clean finish is a restart.
            case "$TRAIN_STATUS" in
                0)
                    echo "$(date): Training finished its iteration budget. Restarting..."
                    start_training
                    ;;
                3)
                    echo "$(date): PLATEAU STOP - the gauge says this run has stopped improving."
                    echo "$(date): Not restarting. See polyfish-rs/ladder.json."
                    exit 0
                    ;;
                *)
                    echo "$(date): Training ABORTED (exit $TRAIN_STATUS). Not restarting." >&2
                    echo "$(date): See polyfish-rs/session.log - a failed gauge reading is fatal by design." >&2
                    exit 1
                    ;;
            esac
        fi
    else
        if [[ -n "$TRAIN_PID" ]]; then
            halt_training "Time window ended."
        fi
    fi

    sleep $CHECK_INTERVAL
done

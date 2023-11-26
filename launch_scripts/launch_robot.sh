#!/bin/bash

# Automated running all hardware nodes
# use bash command to make a file  ~/.tmux.conf and add lines to print pane name
# if [ ! -f ~/.tmux.conf ]; then
# add lines to tmux config to print pane name
# touch ~/.tmux.conf
# echo "set-option -g pane-border-status top" >> ~/.tmux.conf
# echo "set -g pane-border-format '[#[fg=white]#{?pane_active,#[bold],} #P #T #[fg=default,nobold]]'" >> ~/.tmux.conf

SESSION_NAME="Alfred Deployment Stack!"

if tmux has-session -t "$SESSION_NAME" >/dev/null 2>&1; then
    # If session exists, attach to it
    tmux attach-session -t "$SESSION_NAME"
else
    # Launch tmux session
    rosnode kill --all
    tmux new-session -d -s autonomy
    tmux source-file ~/ws/launch_scripts/.tmux-conf

    #split window into 3x3 grid
    tmux split-window -v            # splits into two windows vertically (top-bottom)
    tmux split-window -v            # splits into three windows vertically
    # tmux split-window -v
    tmux select-pane -t 0           # selects pane 0
    tmux split-window -h            # splits that into two windows horizontally (left-right)
    tmux select-pane -t 2
    tmux split-window -h
    tmux select-pane -t 4
    tmux split-window -h
    # tmux select-pane -t 6
    

    # To kill session --> type "Ctrl + B, then Shift + :, then type kill-session"
    # Run commands in each pane (add sleeps to wait for roscore to start)
    # tmux send-keys -t 0 "roscore" C-m
    # tmux send-keys -t 6 "sleep 1 && roslaunch alfred_core driver.launch" C-m
    # tmux send-keys -t 7 "sleep 1 && roslaunch alfred_core perception_robot_tuned.launch" C-m

    tmux send-keys -t 0 "sleep 2 && roslaunch interface_manager interface_manager.launch " C-m
    tmux send-keys -t 1 "sleep 4 && roslaunch mission_planner mission_planner.launch" C-m
    tmux send-keys -t 2 "sleep 6 && roslaunch alfred_navigation navigation.launch" C-m
    tmux send-keys -t 3 "sleep 18 && roslaunch manipulation robot_manipulation.launch" C-m
    tmux send-keys -t 4 "sleep 21 && roslaunch telepresence telepresence_manager.launch" C-m
    tmux send-keys -t 5 "htop" C-m

    # tmux send-keys -t 1 "sleep 10 && roslaunch alfred_core perception_robot_tuned.launch" C-m

    # Attach to tmux session
    tmux attach-session -t autonomy
fi
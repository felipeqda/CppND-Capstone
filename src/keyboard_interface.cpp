#include "keyboard_interface.h"

bool keyboard_interface(int & delay_control, const int default_delay, bool verbose){
    int key_code = cv::waitKey(delay_control);  // integer waiting time in ms
    // if(key_code != -1 && verbose) std::cout << "you pressed: " << key_code << std::endl;
    
    switch (key_code) {
    case keys::esc:
        // esc press will quit program
        if(verbose) std::cout << "ESC pressed: Quitting program!" << std::endl;            
        return true;
        break;

    case keys::espace:
        // toggle zero delay, which will block until next key press (pressing espace will pause/carry on)  
        delay_control = (delay_control==default_delay) ? 0: default_delay;
        return false;
        break;

    case keys::enter:
        // set zero delay, which will block until next key press (pressing enter will step through frames) 
        delay_control = 0;
        return false;
        break;

    default:
        // other key presses will resume execution
        delay_control = default_delay;
        return false;
        break;
    }
};
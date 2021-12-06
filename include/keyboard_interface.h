#ifndef KBI_H
#define KBI_H

#include <iostream>
#include <opencv2/core.hpp>   
#include <opencv2/highgui.hpp>  // OpenCV window I/O

// key code aliases (cf. https://www.c-sharpcorner.com/blogs/ascii-key-code-value1)
namespace keys{
    constexpr int esc    = 27;
    constexpr int espace = 32;
    constexpr int enter  = 13;
};

// what to do on key-press ==> return true on quit loop
bool keyboard_interface(int & delay_control, const int default_delay, bool verbose);

#endif
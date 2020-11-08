#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include <string>
#include <iostream>

namespace fs = std::experimental::filesystem;

int main() {
  fs::directory_iterator itr(fs::current_path());
  while (itr != fs::end(itr)) {
    const fs::directory_entry& entry = *itr;
    std::cout << entry.path() << std::endl;
    itr++;
  }
}


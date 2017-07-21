#ifndef HELPER_H
#define HELPER_H

#include <CL/cl.hpp>
#include <iostream>
#include <string>
#include <fstream>

inline std::string readKernel(const std::string &fileName)
{
  std::ifstream ifs(fileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

  std::ifstream::pos_type fileSize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  std::vector<char> bytes(fileSize);
  ifs.read(&bytes[0], fileSize);

  return std::string(&bytes[0], fileSize);
}

#endif // HELPER_H

/*
 * Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation (CIAN)
 *
 */
#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <ctime>

inline std::string getCurrentDateTime(std::string s)
{
  time_t now = time(0);
  struct tm  tstruct;
  char  buf[80];
  tstruct = *localtime(&now);
  if(s=="now")
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);
  else if(s=="date")
    strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);
  return std::string(buf);
};

inline void writeToLogFile(std::string prefix, std::string logMsg)
{
  std::string filePath = prefix + getCurrentDateTime("date") + ".txt";
  //std::string now = getCurrentDateTime("now");
  std::ofstream ofs(filePath, std::ios_base::out | std::ios_base::app );
  ofs << logMsg << "\n";
  ofs.close();
}

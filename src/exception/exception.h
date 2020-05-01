//
// Created by gs1010 on 01/05/20.
//

#ifndef CMPROJECT_SRC_EXCEPTION_EXCEPTION_H_
#define CMPROJECT_SRC_EXCEPTION_EXCEPTION_H_

#include <exception>
#include <string>
class Exception : public std::exception {
  std::string _msg;
 public:
  Exception(const std::string &msg) : _msg(msg) {}

  virtual const char *what() const noexcept override {
    return _msg.c_str();
  }
};
#endif //CMPROJECT_SRC_EXCEPTION_EXCEPTION_H_

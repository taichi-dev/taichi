/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yu Fang <squarefk@gmail.com>
                  2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#define _WEBSOCKETPP_CPP11_THREAD_
#define ASIO_STANDALONE

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <cstdlib>
#include <thread>
#include <set>
#include <mutex>
#include <vector>
#include <taichi/util.h>
#include <taichi/math/array.h>
#include <taichi/visualization/pakua.h>

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;
using websocketpp::connection_hdl;

typedef websocketpp::server<websocketpp::config::asio> server;
typedef std::set<connection_hdl, std::owner_less<connection_hdl>> con_list;

// pull out the type of messages sent by our config
typedef server::message_ptr message_ptr;
typedef std::set<connection_hdl, std::owner_less<connection_hdl>> con_list;

TC_NAMESPACE_BEGIN

using Vector = VectorND<3, real>;

class WebglPakuaServer : public Task {
  std::string output_path;
  server pakua_server;
  con_list monitor_connections;
  con_list taichi_connections;
  std::thread th;

  void on_open(connection_hdl hdl) {}

  void on_close(connection_hdl hdl) {
    monitor_connections.erase(hdl);
    taichi_connections.erase(hdl);
    printf("There are %lu monitors.\n", monitor_connections.size());
    printf("There are %lu taichi clients.\n", taichi_connections.size());
  }

  void on_message(connection_hdl hdl, message_ptr msg) {
    if (msg->get_payload() == std::string("monitor")) {
      monitor_connections.insert(hdl);
      printf("There are %lu monitors.\n", monitor_connections.size());
    } else if (msg->get_payload() == std::string("monitor")) {
      taichi_connections.insert(hdl);
      printf("There are %lu taichi clients.\n", taichi_connections.size());
    } else if (msg->get_payload().substr(0, 16) ==
               std::string("frame_directory ")) {
      output_path = msg->get_payload().substr(16);
    } else if (monitor_connections.find(hdl) != monitor_connections.end()) {
      unsigned char const *str =
          reinterpret_cast<unsigned char const *>(msg->get_payload().c_str());
      Vector2i res;
      res[0] = reinterpret_cast<uint16 const *>(str)[0];
      res[1] = reinterpret_cast<uint16 const *>(str)[1];
      P(res);
      if (res[0] == 0) {
        // Nothing.
        return;
      }

      str += 4;

      // '* 4' here for RGBA
      if (msg->get_payload().size() != res[0] * res[1] * 4 + 4) {
        printf(
            "Warning: image data size does not match resolution! Data "
            "ignored.\n");
        P(msg->get_payload().size());
        P(res[0] * res[1] * 4 + 4);
        return;
      }

      Array2D<Vector3> img(res);
      for (int i = 0; i < res[0]; i++) {
        int r, g, b;
        for (int j = 0; j < res[1]; j++) {
          int r = str[(j * res[0] + i) * 4];
          int g = str[(j * res[0] + i) * 4 + 1];
          int b = str[(j * res[0] + i) * 4 + 2];
          img[i][res[1] - j - 1] = Vector(r, g, b) * (1 / 255.0f);
        }
      }
      img.write(output_path);
    } else {
      for (auto &connection : monitor_connections) {
        pakua_server.send(connection, msg->get_payload(), msg->get_opcode());
      }
    }
  }

  void run() override {
    th = std::thread([]() {
      std::cout << "*** Server starting at http://localhost:1116 ***\n"
                << std::endl;
      std::string dir = std::string(std::getenv("TAICHI_ROOT_DIR")) +
                        std::string("taichi/pakua/frontend");
      chdir(dir.c_str());
      system("python -m SimpleHTTPServer 1116");
    });
    int port = 9563;
    try {
      // Set logging settings
      pakua_server.set_access_channels(websocketpp::log::alevel::all);
      pakua_server.clear_access_channels(
          websocketpp::log::alevel::frame_payload);
      pakua_server.init_asio();
      pakua_server.set_open_handler(
          bind(&WebglPakuaServer::on_open, this, ::_1));
      pakua_server.set_close_handler(
          bind(&WebglPakuaServer::on_close, this, ::_1));
      pakua_server.set_message_handler(
          bind(&WebglPakuaServer::on_message, this, ::_1, ::_2));
      pakua_server.listen(port);
      pakua_server.start_accept();
      std::cout << "Running ..." << std::endl;
      pakua_server.run();
    } catch (websocketpp::exception const &e) {
      std::cout << e.what() << std::endl;
    } catch (...) {
      std::cout << "other exception" << std::endl;
    }
  }
};

TC_IMPLEMENTATION(Task, WebglPakuaServer, "pakua_server");

TC_NAMESPACE_END
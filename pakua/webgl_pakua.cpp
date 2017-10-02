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
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <thread>
#include <set>
#include <mutex>
#include <vector>
#include <taichi/visualization/pakua.h>
#include <taichi/visualization/image_buffer.h>

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;
using websocketpp::connection_hdl;

typedef websocketpp::client<websocketpp::config::asio_client> asio_client;
typedef websocketpp::config::asio_client::message_type::ptr message_ptr;

TC_NAMESPACE_BEGIN

// implement websocket client
class WebglPakua : public Pakua {
  asio_client pakua_client;
  websocketpp::lib::shared_ptr<websocketpp::lib::thread> client;
  websocketpp::lib::error_code echo_ec;

  std::vector<real> point_buffer;
  std::vector<real> line_buffer;
  std::vector<real> triangle_buffer;
  std::vector<real> resolution_buffer;
  websocketpp::connection_hdl data_hdl;
  int frame_count;
  std::string frame_directory;

#define CHECK_EC                                      \
  if (echo_ec) {                                      \
    printf("Connection initialization error : %s.\n", \
           echo_ec.message().c_str());                \
    assert(false);                                    \
  }

 public:
  ~WebglPakua() {
    pakua_client.stop_perpetual();
    pakua_client.close(data_hdl, websocketpp::close::status::going_away, "",
                       echo_ec);
    CHECK_EC
    client->join();
  }

  void initialize(const Config &config) override {
    Pakua::initialize(config);
    frame_directory = config.get_string("frame_directory");
    frame_count = 0;
    int port = config.get_int("port");
    pakua_client.clear_access_channels(websocketpp::log::alevel::frame_header);
    pakua_client.clear_access_channels(websocketpp::log::alevel::frame_payload);
    pakua_client.init_asio();
    pakua_client.start_perpetual();
    auto th = new websocketpp::lib::thread(&asio_client::run, &pakua_client);
    client.reset(th);
    printf("WebGL Pakua Client runs in a new thread.\n");
    std::string uri = std::string("ws://localhost:") + std::to_string(port);

    // establish data connection
    asio_client::connection_ptr data_con =
        pakua_client.get_connection(uri.c_str(), echo_ec);
    CHECK_EC
    data_hdl = data_con->get_handle();
    pakua_client.connect(data_con);
    pakua_client.set_message_handler(
        bind(&WebglPakua::on_message, this, ::_1, ::_2));
    set_resolution(Vector2i(1024, 1024));
  }

  void set_resolution(Vector2i res) override {
    resolution_buffer.resize(2);
    resolution_buffer[0] = res[0];
    resolution_buffer[1] = res[1];
  }

  void on_message(connection_hdl hdl, message_ptr msg) {
    if (msg->get_payload() == std::string("screen")) {
      P("screen")
    } else if (msg->get_payload() == std::string("taichi")) {
    }
  }

  void add_point(Vector pos, Vector color, real size = 1.0f) override {
    for (int i = 0; i < 3; i++)
      point_buffer.push_back(pos[i]);
    for (int i = 0; i < 3; i++)
      point_buffer.push_back(color[i]);
    point_buffer.push_back(size);
  }

  void add_line(const std::vector<Vector> &pos_v,
                const std::vector<Vector> &color_v,
                real width = 1.0f) override {
    int number = (int)pos_v.size();
    for (int i = 0; i < number; i++) {
      for (int j = 0; j < 3; j++)
        line_buffer.push_back(pos_v[i][j]);
      for (int j = 0; j < 3; j++)
        line_buffer.push_back(color_v[i][j]);
      line_buffer.push_back(width);
    }
  }

  void add_triangle(const std::vector<Vector> &pos_v,
                    const std::vector<Vector> &color_v) override {
    int number = (int)pos_v.size();
    for (int i = 0; i < number; i++) {
      for (int j = 0; j < 3; j++)
        triangle_buffer.push_back(pos_v[i][j]);
      for (int j = 0; j < 3; j++)
        triangle_buffer.push_back(color_v[i][j]);
    }
  }

  void start() override {
    std::string frame_count_str = std::to_string(frame_count);
    std::string output_path = std::string("frame_directory ") +
                              frame_directory + std::string("/") +
                              std::string(5 - frame_count_str.size(), '0') +
                              frame_count_str + std::string(".png");
    pakua_client.send(data_hdl, output_path, websocketpp::frame::opcode::text,
                      echo_ec);
    CHECK_EC
  }

  void finish() override {
    auto send_single_kind = [&](const std::string &kind,
                                std::vector<float> &buffer) {
      pakua_client.send(data_hdl, kind, websocketpp::frame::opcode::text,
                        echo_ec);
      CHECK_EC
      pakua_client.send(data_hdl, &buffer[0], buffer.size() * sizeof(real),
                        websocketpp::frame::opcode::binary, echo_ec);
      CHECK_EC
      buffer.clear();
    };
    std::vector<real> frame_id_buffer;
    frame_id_buffer.push_back(frame_count);
    send_single_kind("point", point_buffer);
    send_single_kind("line", line_buffer);
    send_single_kind("triangle", triangle_buffer);
    send_single_kind("resolution", resolution_buffer);
    send_single_kind("frame_id", frame_id_buffer);

    frame_count += 1;
  }

  Array2D<Vector3> screenshot() {
    pakua_client.send(data_hdl, "screenshot", websocketpp::frame::opcode::text,
                      echo_ec);
    CHECK_EC
    return Array2D<Vector3>();
  }

#undef CHECK_EC
};

TC_IMPLEMENTATION(Pakua, WebglPakua, "webgl");

TC_NAMESPACE_END

/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/dynamics/simulation.h>
#include <taichi/visualization/particle_visualization.h>
#include <taichi/visual/texture.h>

TC_NAMESPACE_BEGIN

class BarnesHutSummation {
 public:
  struct Particle {
    Vector3 position;
    real mass;

    Particle() {
      mass = 0.0_f;
      position = Vector3(0.0_f);
    }

    Particle(const Vector3 &position, real mass)
        : position(position), mass(mass) {}

    Particle operator+(const Particle &o) const {
      Particle ret;
      ret.mass = mass + o.mass;
      ret.position = (position * mass + o.position * o.mass) / ret.mass;
      CV(ret.position);
      CV(ret.mass);
      return ret;
    }
  };

 protected:
  struct Node {
    Particle p;
    int children[8];  // We do not use pointer here to save memory bandwidth(64
                      // bit v.s. 32 bit)
    // There are so many ways to optimize and I'll consider them later...
    Vector3i bounds[2];  // TODO: this is quite brute-force...
    Node() {
      memset(children, 0, sizeof(children));
      p = Particle();
    }

    bool is_leaf() {
      for (int i = 0; i < 8; i++) {
        if (children[i] != 0) {
          return false;
        }
      }
      return p.mass > 0;
    }
  };

  real resolution, inv_resolution;
  int total_levels;
  int margin;

  std::vector<Node> nodes;
  int node_end;
  Vector3 lower_corner;

  Vector3i get_coord(const Vector3 &position) {
    Vector3i u;
    Vector3 t = (position - lower_corner) * inv_resolution;
    for (int i = 0; i < 3; i++) {
      u[i] = int(t[i]);
    }
    return u;
  }

  int get_child_index(const Vector3i &u, int level) {
    int ret = 0;
    for (int i = 0; i < 3; i++) {
      ret += ((u[i] >> (total_levels - level - 1)) & 1) << i;
    }
    return ret;
  }

  void summarize(int t) {
    Node &node = nodes[t];
    if (node.is_leaf()) {
      Vector3i u = get_coord(node.p.position);
      node.bounds[0] = u - Vector3i(margin);
      node.bounds[1] = u + Vector3i(margin);
      return;
    }
    real mass = 0.0_f;
    Vector3 total_position(0.0_f);
    node.bounds[0] = Vector3i(std::numeric_limits<int>::max());
    node.bounds[1] = Vector3i(std::numeric_limits<int>::min());
    for (int c = 0; c < 8; c++) {
      if (node.children[c]) {
        summarize(nodes[t].children[c]);
        const Node &ch = nodes[node.children[c]];
        mass += ch.p.mass;
        total_position += ch.p.mass * ch.p.position;
        for (int i = 0; i < 3; i++) {
          node.bounds[0][i] = std::min(node.bounds[0][i], ch.bounds[0][i]);
          node.bounds[1][i] = std::max(node.bounds[1][i], ch.bounds[1][i]);
        }
      }
    }
    total_position *= 1.0_f / mass;
    if (total_position.abnormal()) {
      P(mass);
      for (int i = 0; i < 8; i++) {
        P(i);
        P(node.children[i]);
        P(nodes[node.children[i]].p.position);
        P(nodes[node.children[i]].p.mass);
      }
    }
    CV(total_position);
    node.p = Particle(total_position, mass);
  }

  int create_child(int t, int child_index) {
    return create_child(t, child_index, Particle(Vector3(0.0_f), 0.0_f));
  }

  int create_child(int t, int child_index, const Particle &p) {
    int nt = get_new_node();
    nodes[t].children[child_index] = nt;
    nodes[nt].p = p;
    return nt;
  }

  int get_new_node() {
    nodes[node_end] = Node();
    return node_end++;
  }

 public:
  // We do not evaluate the weighted average of position and mass on the fly
  // for efficiency and accuracy
  void initialize(real resolution,
                  real margin_real,
                  const std::vector<Particle> &particles) {
    this->resolution = resolution;
    this->inv_resolution = 1.0_f / resolution;
    this->margin = (int)std::ceil(margin_real * inv_resolution);
    assert(particles.size() != 0);
    Vector3 lower(1e30f);
    Vector3 upper(-1e30f);
    for (auto &p : particles) {
      for (int k = 0; k < 3; k++) {
        lower[k] = std::min(lower[k], p.position[k]);
        upper[k] = std::max(upper[k], p.position[k]);
      }
      // P(p.position);
    }
    lower_corner = lower;
    int intervals = (int)std::ceil((upper - lower).max() / resolution);
    total_levels = 0;
    for (int i = 1; i < intervals; i *= 2, total_levels++)
      ;
    // We do not use the 0th node...
    node_end = 1;
    nodes.clear();
    nodes.resize(particles.size() * 2);
    int root = get_new_node();
    // Make sure that one leaf node contains only one particle.
    // Unless particles are too close and thereby merged.
    for (auto &p : particles) {
      if (p.mass == 0) {
        continue;
      }
      Vector3i u = get_coord(p.position);
      int t = root;
      if (nodes[t].is_leaf()) {
        // First node
        nodes[t].p = p;
        continue;
      }
      // Traverse down until there's no way...
      int k = 0;
      int cp;
      for (; k < total_levels; k++) {
        cp = get_child_index(u, k);
        if (nodes[t].children[cp] != 0) {
          t = nodes[t].children[cp];
        } else {
          break;
        }
      }
      if (nodes[t].is_leaf()) {
        // Leaf node, containing one particle q
        // Split the node until p and q belong to different children.
        Particle q = nodes[t].p;
        nodes[t].p = Particle();
        Vector3i v = get_coord(q.position);
        int cq = get_child_index(v, k);
        while (cp == cq && k < total_levels) {
          t = create_child(t, cp);
          k++;
          cp = get_child_index(u, k);
          cq = get_child_index(v, k);
        }
        if (k == total_levels) {
          // We have to merge two particles since they are too close...
          q = p + q;
          create_child(t, cp, q);
        } else {
          nodes[t].p = Particle();
          create_child(t, cp, p);
          create_child(t, cq, q);
        }
      } else {
        // Non-leaf node, simply create a child.
        create_child(t, cp, p);
      }
    }
    P(node_end);
    summarize(root);
  }

  /*
  template<typename T>
  Vector3 summation(const Particle &p, const T &func) {
      // TODO: fine level
      // TODO: only one particle?
      int t = 1;
      Vector3 ret(0.0_f);
      Vector3 u = get_coord(p.position);
      for (int k = 0; k < total_levels; k++) {
          int cp = get_child_index(u, k);
          for (int c = 0; c < 8; c++) {
              if (c != cp && nodes[t].children[c]) {
                  const Node &n = nodes[nodes[t].children[c]];
                  auto tmp = func(p, n.p);
                  ret += tmp;
              }
          }
          t = nodes[t].children[cp];
          if (t == 0) {
              break;
          }
      }
      return ret;
  }
  */

  template <typename T>
  Vector3 summation(int t, const Particle &p, const T &func) {
    const Node &node = nodes[t];
    if (nodes[t].is_leaf()) {
      return func(p, node.p);
    }
    Vector3 ret(0.0_f);
    Vector3i u = get_coord(p.position);
    for (int c = 0; c < 8; c++) {
      if (node.children[c]) {
        const Node &ch = nodes[node.children[c]];
        if (ch.bounds[0][0] <= u[0] && u[0] <= ch.bounds[1][0] &&
            ch.bounds[0][1] <= u[1] && u[1] <= ch.bounds[1][1] &&
            ch.bounds[0][2] <= u[2] && u[2] <= ch.bounds[1][2]) {
          ret += summation(node.children[c], p, func);
        } else {
          // Coarse summation
          ret += func(p, ch.p);
        }
      }
    }
    return ret;
  }

  void print_tree(int t, int level) {
    for (int i = 0; i < level; i++) {
      printf("  ");
    }
    const Particle &p = nodes[t].p;
    printf("p (%f, %f, %f) m %f ", p.position.x, p.position.y, p.position.z,
           p.mass);
    printf("(%d, %d, %d) (%d, %d, %d)\n", nodes[t].bounds[0][0],
           nodes[t].bounds[0][1], nodes[t].bounds[0][2], nodes[t].bounds[1][0],
           nodes[t].bounds[1][1], nodes[t].bounds[1][2]);
    for (int c = 0; c < 8; c++) {
      if (nodes[t].children[c] != 0) {
        print_tree(nodes[t].children[c], level + 1);
      }
    }
  }
};

class NBody : public Simulation3D {
  struct Particle {
    Vector3 position, velocity, color;

    Particle(const Vector3 &position,
             const Vector3 &velocity,
             const Vector3 &color)
        : position(position), velocity(velocity), color(color) {}
  };

 protected:
  real gravitation;
  std::shared_ptr<Texture> velocity_field;
  std::vector<Particle> particles;
  BarnesHutSummation bhs;
  real delta_t;

 public:
  virtual void initialize(const Config &config) override {
    Simulation3D::initialize(config);
    int num_particles = config.get<int>("num_particles");
    particles.reserve(num_particles);
    gravitation = config.get<real>("gravitation");
    delta_t = config.get<real>("delta_t");
    real vel_scale = config.get<real>("vel_scale");
    for (int i = 0; i < num_particles; i++) {
      Vector3 p(rand(), rand(), rand());
      Vector3 v = Vector3(p.y, p.z, p.x) - Vector3(0.5f);
      v *= vel_scale;
      Vector3 c(0.5, 0.7, 0.4);
      particles.push_back(Particle(p, v, c));
    }
  }

  std::vector<RenderParticle> get_render_particles() const override {
    std::vector<RenderParticle> render_particles;
    render_particles.reserve(particles.size());
    for (auto &p : particles) {
      render_particles.push_back(
          RenderParticle(p.position - Vector3(0.5f), p.color));
    }
    return render_particles;
  }

  void substep(real dt) {
    using BHP = BarnesHutSummation::Particle;
    std::vector<BHP> bhps;
    bhps.reserve(particles.size());
    for (auto &p : particles) {
      bhps.push_back(BHP(p.position, 1.0_f));
    }

    bhs.initialize(1e-4_f, 1e-3_f, bhps);
    // bhs.print_tree(1, 0);

    auto f = [](const BHP &p, const BHP &q) {
      CV(p.position);
      CV(q.position);
      Vector3 d = p.position - q.position;
      real dist2 = dot(d, d);
      dist2 += 1e-4_f;
      CV(d);
      d *= p.mass * q.mass / (dist2 * sqrt(dist2));
      CV(p.mass);
      CV(q.mass);
      CV(dist2);
      CV(d);
      return d;
    };
    if (gravitation != 0) {
      real max_err = -1;
      ThreadedTaskManager::run((int)particles.size(), num_threads, [&](int i) {
        auto &p = particles[i];
        /*
        Vector3 total_f(0.0_f);
        for (int j = 0; j < (int)particles.size(); j++) {
            auto &q = particles[j];
            total_f += f(BHP(p.position, 1.0_f), BHP(q.position, 1.0_f));
        }

        auto err = length(total_f_bhs - total_f) / length(total_f);
        max_err = std::max(err, max_err);

        P(total_f);
        P(total_f_bhs);
        P(err);
        */
        Vector3 total_f_bhs = bhs.summation(1, BHP(p.position, 1.0_f), f);
        CV(total_f_bhs);
        particles[i].velocity += total_f_bhs * gravitation * dt;
        CV(particles[i].velocity);
        CV(total_f_bhs);
      });
      // P(max_err);
    }
    for (auto &p : particles) {
      p.position += dt * p.velocity;
    }
    current_t += dt;
  }

  virtual void step(real dt) override {
    int steps = (int)std::ceil(dt / delta_t);
    for (int i = 0; i < steps; i++) {
      substep(dt / steps);
    }
  }
};

TC_IMPLEMENTATION(Simulation3D, NBody, "nbody");

TC_NAMESPACE_END

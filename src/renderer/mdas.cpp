/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2016 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/renderer.h>

TC_NAMESPACE_BEGIN

#define TC_MDAS_MAX_DIM 8

    struct Node {
        real bounds[TC_MDAS_MAX_DIM][2];
        Node *ch[2];
        real split;

    };

    class KDTree {
    public:

        KDTree() {

        }

        void insert_new_sample() {

        }

    };

    class MDAS : public Renderer {
    protected:
        int samples_per_stage;
    public:
        virtual void initialize(const Config &config) override {
            samples_per_stage = config.get("samples_per_stages", 128);

        }

        void render_stage() override {

        }

        Array2D<Vector3> get_reconstruction() {

        }

        virtual Array2D<Vector3> get_output() override {
            return get_reconstruction();
        }

    protected:

    };

    TC_IMPLEMENTATION(Renderer, MDAS, "mdas");

TC_NAMESPACE_END


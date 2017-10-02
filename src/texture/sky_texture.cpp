/*******************************************************************************
    Taichi - Physically based Computer Graphics Library

    Copyright (c) 2017 Yuanming Hu <yuanmhu@gmail.com>

    All rights reserved. Use of this source code is governed by
    the MIT license as written in the LICENSE file.
*******************************************************************************/

#include <taichi/visual/texture.h>

TC_NAMESPACE_BEGIN

// TODO: generalize
Vector3 exp(Vector3 a) {
  return Vector3(std::exp(a[0]), std::exp(a[1]), std::exp(a[2]));
}

Vector3 pow(Vector3 a, Vector3 b) {
  return Vector3(std::pow(a[0], b[0]), std::pow(a[1], b[1]),
                 std::pow(a[2], b[2]));
}

Vector3 lerp(Vector3 a, Vector3 b, Vector3 x) {
  return a * (Vector3(1.0) - x) + b * x;
}

/**
 * Based on "A Practical Analytic Model for Daylight"
 * aka The Preetham Model, the de facto standard analytic skydome model
 * http://www.cs.utah.edu/~shirley/papers/sunsky/sunsky.pdf
 *
 * First implemented by Simon Wallner
 * http://www.simonwallner.at/projects/atmospheric-scattering
 *
 * Improved by Martin Upitis
 * http://blenderartists.org/forum/showthread.php?245954-preethams-sky-impementation-HDR
 *
 * Three.js integration by zz85 http://twitter.com/blurspline
 *
 * Taichi integration by Yuanming Hu <yuanmhu@gmail.com>
*/

real smoothstep(real edge0, real edge1, real x) {
  real t = clamp((x - edge0) / (edge1 - edge0), 0.0_f, 1.0_f);
  return t * t * (3.0f - 2.0f * t);
}

class SkyTexture final : public Texture {
 private:
  Vector4 val;

  // Variables
  real luminance;
  real turbidity;
  real rayleigh;
  real mie_coefficient;
  real mie_directional_g;
  Vector3 sun_position;

  // constants for atmospheric scattering
  static constexpr real e = 2.718281828459f;

  static const Vector3 up;
  // wavelength of used primaries, according to preetham
  static const Vector3 lambda;

  // this pre-calcuation replaces older TotalRayleigh(Vector3 lambda) function:
  // (8.0 * pow(pi, 3.0) * pow(pow(n, 2.0) - 1.0, 2.0) * (6.0 + 3.0 * pn)) /
  // (3.0 * N * pow(lambda, Vector3(4.0)) * (6.0 - 7.0 * pn))
  static const Vector3 totalRayleigh;

  // mie stuff
  // K coefficient for the primaries
  static constexpr real v = 4.0;

  static const Vector3 K;
  // MieConst = pi * pow( ( 2.0 * pi ) / lambda, Vector3( v - 2.0 ) ) * K
  static const Vector3 MieConst;
  // earth shadow hack
  // cutoffAngle = pi / 1.95;
  static constexpr real cutoffAngle = 1.6110731556870734f;
  static constexpr real steepness = 1.5f;
  static constexpr real EE = 1000.0_f;

  static real sunIntensity(real zenithAngleCos) {
    zenithAngleCos = clamp(zenithAngleCos, -1.0_f, 1.0_f);
    return EE *
           std::max(
               0.0_f,
               1.0_f - std::pow(e,
                               -((cutoffAngle - std::acos(zenithAngleCos)) /
                                 steepness)));
  }

  static Vector3 totalMie(real T) {
    real c = (0.2f * T) * 10E-18f;
    return 0.434f * c * MieConst;
  }

  Vector3 cameraPos;

  static constexpr real n = 1.0003;  // refractive index of air
  static constexpr real N =
      2.545E25;  // number of molecules per unit volume for air at
  // 288.15K and 1013mb (sea level -45 celsius)

  // optical length at zenith for molecules
  static constexpr real rayleighZenithLength = 8.4E3;
  static constexpr real mieZenithLength = 1.25E3;
  // 66 arc seconds -> degrees, and the cosine of that
  static constexpr real sunAngularDiameterCos = 0.9999566769f;

  // 3.0 / ( 16.0 * pi )
  static constexpr real THREE_OVER_SIXTEENPI = 0.05968310f;
  // 1.0 / ( 4.0 * pi )
  static constexpr real ONE_OVER_FOURPI = 0.07957747f;

  static real rayleighPhase(real cosTheta) {
    return THREE_OVER_SIXTEENPI * (1.0_f + std::pow(cosTheta, 2.0f));
  }

  static real hgPhase(real cosTheta, real g) {
    real g2 = std::pow(g, 2.0f);
    real inverse = 1.0_f / pow(1.0_f - 2.0f * g * cosTheta + g2, 1.5f);
    return ONE_OVER_FOURPI * ((1.0_f - g2) * inverse);
  }

 public:
  void initialize(const Config &config) override {
    Texture::initialize(config);
    TC_LOAD_CONFIG(luminance, 1.0_f);
    TC_LOAD_CONFIG(turbidity, 10.0_f);
    TC_LOAD_CONFIG(rayleigh, 2.0f);
    TC_LOAD_CONFIG(mie_coefficient, 0.005f);
    TC_LOAD_CONFIG(mie_directional_g, 0.8f);
    TC_LOAD_CONFIG(cameraPos, Vector3(0.0_f));
    real theta = config.get("direction", 0.0_f) * 2 * pi;
    real phi = (config.get("height", 0.0_f) - 0.5) * pi;
    sun_position =
        Vector3(cos(theta) * cos(phi), sin(phi), sin(theta) * cos(phi));
  }

  virtual Vector4 sample(const Vector3 &coord) const override {
    // TODO: what is position?
    Vector3 vWorldPosition;
    Vector3 vSunDirection;
    real vSunfade;
    Vector3 vBetaR;
    Vector3 vBetaM;
    real vSunE;

    real luminance;
    real mieDirectionalG;

    real theta_d = coord.x * 2 * pi, phi_d = (coord.y - 0.5f) * pi;
    vWorldPosition = Vector3(cos(theta_d) * cos(phi_d), sin(phi_d),
                             sin(theta_d) * cos(phi_d));
    vSunDirection = normalized(sun_position);

    vSunE = sunIntensity(dot(vSunDirection, up));

    vSunfade =
        1.0_f - clamp(1.0_f - std::exp((sun_position.y / 450000.0_f)), 0.0_f, 1.0_f);

    real rayleighCoefficient = rayleigh - (1.0_f * (1.0_f - vSunfade));

    // extinction (absorbtion + out scattering)
    // rayleigh coefficients
    vBetaR = totalRayleigh * rayleighCoefficient;

    // mie coefficients
    vBetaM = totalMie(turbidity) * mie_coefficient;
    real zenithAngle = std::acos(
        std::max(0.0_f, dot(up, normalize(vWorldPosition - cameraPos))));
    real inverse =
        1.0_f /
        (std::cos(zenithAngle) +
         0.15f * std::pow(93.885f - ((zenithAngle * 180.0_f) / pi), -1.253f));
    real sR = rayleighZenithLength * inverse;
    real sM = mieZenithLength * inverse;

    // combined extinction factor
    Vector3 Fex = exp(-(vBetaR * sR + vBetaM * sM));

    // in scattering
    real cosTheta = dot(normalize(vWorldPosition - cameraPos), vSunDirection);

    real rPhase = rayleighPhase(cosTheta * 0.5f + 0.5f);
    Vector3 betaRTheta = vBetaR * rPhase;

    real mPhase = hgPhase(cosTheta, mieDirectionalG);
    Vector3 betaMTheta = vBetaM * mPhase;

    Vector3 Lin = pow(vSunE * ((betaRTheta + betaMTheta) / (vBetaR + vBetaM)) *
                          (Vector3(1.0_f) - Fex),
                      Vector3(1.5f));
    Lin *=
        lerp(Vector3(1.0_f),
             pow(vSunE * ((betaRTheta + betaMTheta) / (vBetaR + vBetaM)) * Fex,
                 Vector3(1.0_f / 2.0f)),
             clamp(pow(1.0_f - dot(up, vSunDirection), 5.0f), 0.0_f, 1.0_f));

    // nightsky
    Vector3 direction = normalize(vWorldPosition - cameraPos);
    real theta = std::acos(direction.y);  // elevation --> y-axis, [-pi/2, pi/2]
    // NOTE: changed atan to atan2
    real phi = std::atan2(direction.z,
                          direction.x);  // azimuth --> x-axis [-pi/2, pi/2]
    Vector2 uv =
        Vector2(phi, theta) / Vector2(2.0f * pi, pi) + Vector2(0.5f, 0.0_f);
    Vector3 L0 = Vector3(0.1f) * Fex;

    // composition + solar disc
    real sundisk = smoothstep(sunAngularDiameterCos,
                              sunAngularDiameterCos + 0.00002f, cosTheta);
    L0 += (vSunE * 19000.0_f * Fex) * sundisk;

    Vector3 texColor = (Lin + L0) * 0.04f + Vector3(0.0_f, 0.0003f, 0.00075f);

    // Vector3 curr = Uncharted2Tonemap( ( log2( 2.0 / pow( luminance, 4.0 ) ) )
    // * texColor );
    // Vector3 color = curr * whiteScale;

    // Vector3 retColor = pow( color, Vector3( 1.0 / ( 1.2 + ( 1.2 * vSunfade )
    // ) ) );
    return Vector4(texColor, 1.0_f);
  }
};

const Vector3 SkyTexture::up(0.0_f, 1.0_f, 0.0_f);
// wavelength of used primaries, according to preetham
const Vector3 SkyTexture::lambda(680E-9f, 550E-9f, 450E-9f);

// this pre-calcuation replaces older TotalRayleigh(Vector3 lambda) function:
// (8.0 * pow(pi, 3.0) * pow(pow(n, 2.0) - 1.0, 2.0) * (6.0 + 3.0 * pn)) / (3.0
// * N * pow(lambda, Vector3(4.0)) * (6.0 - 7.0 * pn))
const Vector3 SkyTexture::totalRayleigh = Vector3(5.804542996261093E-6_f,
                                                  1.3562911419845635E-5_f,
                                                  3.0265902468824876E-5_f);

// mie stuff
// K coefficient for the primaries
constexpr real v = 4.0;

const Vector3 SkyTexture::K = Vector3(0.686f, 0.678f, 0.666f);
// MieConst = pi * pow( ( 2.0 * pi ) / lambda, Vector3( v - 2.0 ) ) * K
const Vector3 SkyTexture::MieConst = Vector3(1.8399918514433978E14f,
                                             2.7798023919660528E14f,
                                             4.0790479543861094E14f);

TC_IMPLEMENTATION(Texture, SkyTexture, "sky");

TC_NAMESPACE_END

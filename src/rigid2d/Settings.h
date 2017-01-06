#ifndef SETTINGS_H
#define SETTINGS_H

class Settings {

public:
    int velocityIteration;
    int positionIteration;
    int stepIteration;
    int frictionSwitch;
    double gravity;
    bool pause;
    int frameRate;
    void Init() {
        velocityIteration = 40;
        positionIteration = 5;
        stepIteration = 30;
        frictionSwitch = true;
        gravity = 9.8;
        pause = false;
        InitAntTweakBar();
    }
    void InitAntTweakBar();
};

extern Settings settings;

#endif
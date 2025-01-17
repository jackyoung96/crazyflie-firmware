
#include "math3d.h"
#include "stabilizer_types.h"
#include <math.h>
#include "controller_nn.h"
#include "log.h"
#include "param.h"
#include "usec_time.h"
#include "network_evaluate.h"


#define MAX_THRUST 0.15f
// PWM to thrust coefficients
#define A 2.130295e-11f
#define B 1.032633e-6f
#define C 5.484560e-4f
#define M_PIF 3.141592653589793238462643383279502884e+00F

static bool enableBigQuad = false;

static float maxThrustFactor = 1.0f; // TODO: We don't care about max thrust 
static bool relVel = false;
static bool relOmega = false;
static bool relXYZ = false;
static uint16_t freq = 200;

static control_t_n control_n;
static struct mat33 rot;
// static float state_array[18];
static float state_array[22];

static uint32_t usec_eval;

void controllerNNInit(void) {
	control_n.thrust_0 = 0.0f;
	control_n.thrust_1 = 0.0f;
	control_n.thrust_2 = 0.0f;
	control_n.thrust_3 = 0.0f;
}



bool controllerNNTest(void) {
	return true;
}


void controllerNNEnableBigQuad(void)
{
	enableBigQuad = true;
}



// range of action -1 ... 1, need to scale to range 0 .. 1
float scale(float v) {
	return 0.5f * (v + 1);
}


float clip(float v, float min, float max) {
	if (v < min) return min;
	if (v > max) return max;
	return v;
}


void controllerNN(motors_thrust_t *control, 
				  setpoint_t *setpoint, 
				  const sensorData_t *sensors, 
				  const state_t *state, 
				  const uint32_t tick)
{
	// control->enableDirectThrust = true;
	if (!RATE_DO_EXECUTE(/*RATE_100_HZ*/freq, tick)) {
		return;
	}

	// Orientation
	struct quat q = mkquat(state->attitudeQuaternion.x, 
						   state->attitudeQuaternion.y, 
						   state->attitudeQuaternion.z, 
						   state->attitudeQuaternion.w);
	rot = quat2rotmat(q);

	// angular velocity
	float omega_roll = radians(sensors->gyro.x);
	float omega_pitch = radians(sensors->gyro.y);
	float omega_yaw = radians(sensors->gyro.z);

	// the state vector
	// Normalized: pose (/6), vel (/3), angvel (/2pi)
	state_array[0] = (state->position.x - setpoint->position.x);
	state_array[1] = (state->position.y - setpoint->position.y);
	state_array[2] = (state->position.z - setpoint->position.z);

    // rotation matrix
    state_array[3] = rot.m[0][0];
	state_array[4] = rot.m[0][1];
	state_array[5] = rot.m[0][2];
	state_array[6] = rot.m[1][0];
	state_array[7] = rot.m[1][1];
	state_array[8] = rot.m[1][2];
	state_array[9] = rot.m[2][0];
	state_array[10] = rot.m[2][1];
	state_array[11] = rot.m[2][2];

	if (relVel) {
		state_array[12] = (state->velocity.x - setpoint->velocity.x);
		state_array[13] = (state->velocity.y - setpoint->velocity.y);
		state_array[14] = (state->velocity.z - setpoint->velocity.z);
	} else {
		state_array[12] = (state->velocity.x);
		state_array[13] = (state->velocity.y);
		state_array[14] = (state->velocity.z);
	}
	

	if (relXYZ) {
		// rotate pos and vel
		struct vec rot_pos = mvmul(mtranspose(rot), mkvec(state_array[0], state_array[1], state_array[2]));
		struct vec rot_vel = mvmul(mtranspose(rot), mkvec(state_array[12], state_array[13], state_array[14]));

		state_array[0] = rot_pos.x;
		state_array[1] = rot_pos.y;
		state_array[2] = rot_pos.z;

		state_array[12] = rot_vel.x;
		state_array[13] = rot_vel.y;
		state_array[14] = rot_vel.z;
	} 

	if (relOmega) {
		state_array[15] = omega_roll - radians(setpoint->attitudeRate.roll);
		state_array[16] = omega_pitch - radians(setpoint->attitudeRate.pitch);
		state_array[17] = omega_yaw - radians(setpoint->attitudeRate.yaw);
	} else {
		state_array[15] = omega_roll;
		state_array[16] = omega_pitch;
		state_array[17] = omega_yaw;
	}

    // Normalize
    state_array[0] = state_array[0] / 6;
    state_array[1] = state_array[1] / 6;
    state_array[2] = state_array[2] / 6;
    state_array[12] = state_array[12] / 3;
    state_array[13] = state_array[13] / 3;
    state_array[14] = state_array[14] / 3;
    state_array[15] = state_array[15] / (2*M_PIF);
    state_array[16] = state_array[16] / (2*M_PIF);
    state_array[17] = state_array[17] / (2*M_PIF);
	state_array[18] = control_n.thrust_0;
	state_array[19] = control_n.thrust_1;
	state_array[20] = control_n.thrust_2;
	state_array[21] = control_n.thrust_3;


	// run the neural neural network
	uint64_t start = usecTimestamp();
	networkEvaluate(&control_n, state_array);
	usec_eval = (uint32_t) (usecTimestamp() - start);

	// convert thrusts to directly to PWM
	// need to hack the firmware (stablizer.c and power_distribution_stock.c)
	int PWM_0, PWM_1, PWM_2, PWM_3; 
	thrusts2PWM(&control_n, &PWM_0, &PWM_1, &PWM_2, &PWM_3);

	if (setpoint->mode.z == modeDisable) {
		control->m1 = 0;
		control->m2 = 0;
		control->m3 = 0;
		control->m4 = 0;
	} else {
		control->m1 = PWM_0;
		control->m2 = PWM_1;
		control->m3 = PWM_2;
		control->m4 = PWM_3;
	}
}


void thrusts2PWM(control_t_n *control_n, 
	int *PWM_0, int *PWM_1, int *PWM_2, int *PWM_3){

	// scaling and cliping
	if (enableBigQuad) {
		// Big quad => output angular velocity of rotors

		// motor 0
		*PWM_0 = maxThrustFactor * UINT16_MAX * clip(scale(control_n->thrust_0), 0.0, 1.0);
		// motor 1
		*PWM_1 = maxThrustFactor * UINT16_MAX * clip(scale(control_n->thrust_1), 0.0, 1.0);
		// motor
		*PWM_2 = maxThrustFactor * UINT16_MAX * clip(scale(control_n->thrust_2), 0.0, 1.0);
		// motor 3 
		*PWM_3 = maxThrustFactor * UINT16_MAX * clip(scale(control_n->thrust_3), 0.0, 1.0);

	} else {
		// Regular Crazyflie => output thrust directly
		// motor 0
		*PWM_0 = 1.0f * UINT16_MAX * clip(scale(control_n->thrust_0), 0.0, 1.0);
		// motor 1
		*PWM_1 = 1.0f * UINT16_MAX * clip(scale(control_n->thrust_1), 0.0, 1.0);
		// motor
		*PWM_2 = 1.0f * UINT16_MAX * clip(scale(control_n->thrust_2), 0.0, 1.0);
		// motor 3 
		*PWM_3 = 1.0f * UINT16_MAX * clip(scale(control_n->thrust_3), 0.0, 1.0);
	}

}

PARAM_GROUP_START(ctrlNN)
PARAM_ADD(PARAM_FLOAT, max_thrust, &maxThrustFactor)
PARAM_ADD(PARAM_UINT8, rel_vel, &relVel)
PARAM_ADD(PARAM_UINT8, rel_omega, &relOmega)
PARAM_ADD(PARAM_UINT8, rel_xyz, &relXYZ)
PARAM_ADD(PARAM_UINT16, freq, &freq)
PARAM_GROUP_STOP(ctrlNN)

LOG_GROUP_START(ctrlNN)
LOG_ADD(LOG_FLOAT, out0, &control_n.thrust_0)
LOG_ADD(LOG_FLOAT, out1, &control_n.thrust_1)
LOG_ADD(LOG_FLOAT, out2, &control_n.thrust_2)
LOG_ADD(LOG_FLOAT, out3, &control_n.thrust_3)

LOG_ADD(LOG_FLOAT, in0, &state_array[0])
LOG_ADD(LOG_FLOAT, in1, &state_array[1])
LOG_ADD(LOG_FLOAT, in2, &state_array[2])

LOG_ADD(LOG_FLOAT, in3, &state_array[3])
LOG_ADD(LOG_FLOAT, in4, &state_array[4])
LOG_ADD(LOG_FLOAT, in5, &state_array[5])

LOG_ADD(LOG_FLOAT, in15, &state_array[15])
LOG_ADD(LOG_FLOAT, in16, &state_array[16])
LOG_ADD(LOG_FLOAT, in17, &state_array[17])

LOG_ADD(LOG_UINT32, usec_eval, &usec_eval)

LOG_GROUP_STOP(ctrlNN)
#include <jni.h>
#include <fstream>
#include <string>
#include <sstream>
#include <android/log.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <map>

#include "thneedmodel.h"
#include "clutil.h"
#include "timing.h"
#include "json11.hpp"
#include "util.h"
#include <dlfcn.h>
#include "CL/cl.h"

#ifndef USE_PRECOMPILED

/*MY loch*/
#include <jni.h>
#include <map>
#include <vector>
#include <cmath>
#include <string>
#include <android/log.h>
#include "modelv2.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NativeModel", __VA_ARGS__)


enum ConfidenceClass {
    GREEN,
    YELLOW,
    RED
};

using ModelMap = std::map<std::string, std::vector<float>>;

// Helper function to convert std::array to std::vector
template<typename T, size_t N>
std::vector<T> array_to_vector(const std::array<T, N> &arr) {
    return std::vector<T>(arr.begin(), arr.end());
}

// Existing fill_model function (updated for new TRAJECTORY_SIZE and DESIRE_PRED_LEN)
void fill_model(ModelMap &data, ConfidenceClass &confidence, const ModelOutput &net_outputs) {
    static std::array<float, 5> prev_brake_5ms2_probs;
    static std::array<float, 3> prev_brake_3ms2_probs;
    static std::array<float, DISENGAGE_LEN * DISENGAGE_LEN> disengage_buffer;

    const auto &best_plan = net_outputs.plans.get_best_prediction();
    std::vector<float> plan_t(TRAJECTORY_SIZE, NAN);
    plan_t[0] = 0.0f;
    for (int xidx = 1, tidx = 0; xidx < TRAJECTORY_SIZE; xidx++) {
        for (int next_tid = tidx + 1; next_tid < TRAJECTORY_SIZE &&
                                      best_plan.mean[next_tid].position.x <
                                      X_IDXS[xidx]; next_tid++) {
            tidx++;
        }
        if (tidx == TRAJECTORY_SIZE - 1) {
            plan_t[xidx] = T_IDXS[TRAJECTORY_SIZE - 1];
            break;
        }
        float current_x_val = best_plan.mean[tidx].position.x;
        float next_x_val = best_plan.mean[tidx + 1].position.x;
        float p = (X_IDXS[xidx] - current_x_val) / (next_x_val - current_x_val);
        plan_t[xidx] = p * T_IDXS[tidx + 1] + (1.0f - p) * T_IDXS[tidx];
    }

    // Fill plan
    {
        std::vector<float> pos_x(TRAJECTORY_SIZE), pos_y(TRAJECTORY_SIZE), pos_z(TRAJECTORY_SIZE);
        std::vector<float> pos_x_std(TRAJECTORY_SIZE), pos_y_std(TRAJECTORY_SIZE), pos_z_std(
                TRAJECTORY_SIZE);
        std::vector<float> vel_x(TRAJECTORY_SIZE), vel_y(TRAJECTORY_SIZE), vel_z(TRAJECTORY_SIZE);
        std::vector<float> acc_x(TRAJECTORY_SIZE), acc_y(TRAJECTORY_SIZE), acc_z(TRAJECTORY_SIZE);
        std::vector<float> rot_x(TRAJECTORY_SIZE), rot_y(TRAJECTORY_SIZE), rot_z(TRAJECTORY_SIZE);
        std::vector<float> rot_rate_x(TRAJECTORY_SIZE), rot_rate_y(TRAJECTORY_SIZE), rot_rate_z(
                TRAJECTORY_SIZE);

        for (int i = 0; i < TRAJECTORY_SIZE; i++) {
            pos_x[i] = best_plan.mean[i].position.x;
            pos_y[i] = best_plan.mean[i].position.y;
            pos_z[i] = best_plan.mean[i].position.z;
            pos_x_std[i] = std::exp(best_plan.std[i].position.x);
            pos_y_std[i] = std::exp(best_plan.std[i].position.y);
            pos_z_std[i] = std::exp(best_plan.std[i].position.z);
            vel_x[i] = best_plan.mean[i].velocity.x;
            vel_y[i] = best_plan.mean[i].velocity.y;
            vel_z[i] = best_plan.mean[i].velocity.z;
            acc_x[i] = best_plan.mean[i].acceleration.x;
            acc_y[i] = best_plan.mean[i].acceleration.y;
            acc_z[i] = best_plan.mean[i].acceleration.z;
            rot_x[i] = best_plan.mean[i].rotation.x;
            rot_y[i] = best_plan.mean[i].rotation.y;
            rot_z[i] = best_plan.mean[i].rotation.z;
            rot_rate_x[i] = best_plan.mean[i].rotation_rate.x;
            rot_rate_y[i] = best_plan.mean[i].rotation_rate.y;
            rot_rate_z[i] = best_plan.mean[i].rotation_rate.z;
        }

        data["position_t"] = array_to_vector(T_IDXS_FLOAT);
        data["position_x"] = pos_x;
        data["position_y"] = pos_y;
        data["position_z"] = pos_z;
        data["position_x_std"] = pos_x_std;
        data["position_y_std"] = pos_y_std;
        data["position_z_std"] = pos_z_std;

        data["velocity_t"] = array_to_vector(T_IDXS_FLOAT);
        data["velocity_x"] = vel_x;
        data["velocity_y"] = vel_y;
        data["velocity_z"] = vel_z;

        data["acceleration_t"] = array_to_vector(T_IDXS_FLOAT);
        data["acceleration_x"] = acc_x;
        data["acceleration_y"] = acc_y;
        data["acceleration_z"] = acc_z;

        data["orientation_t"] = array_to_vector(T_IDXS_FLOAT);
        data["orientation_x"] = rot_x;
        data["orientation_y"] = rot_y;
        data["orientation_z"] = rot_z;

        data["orientation_rate_t"] = array_to_vector(T_IDXS_FLOAT);
        data["orientation_rate_x"] = rot_rate_x;
        data["orientation_rate_y"] = rot_rate_y;
        data["orientation_rate_z"] = rot_rate_z;
    }

    // Fill lane lines
    {
        std::vector<float> left_far_y(TRAJECTORY_SIZE), left_far_z(TRAJECTORY_SIZE);
        std::vector<float> left_near_y(TRAJECTORY_SIZE), left_near_z(TRAJECTORY_SIZE);
        std::vector<float> right_near_y(TRAJECTORY_SIZE), right_near_z(TRAJECTORY_SIZE);
        std::vector<float> right_far_y(TRAJECTORY_SIZE), right_far_z(TRAJECTORY_SIZE);
        for (int j = 0; j < TRAJECTORY_SIZE; j++) {
            left_far_y[j] = net_outputs.lane_lines.mean.left_far[j].y;
            left_far_z[j] = net_outputs.lane_lines.mean.left_far[j].z;
            left_near_y[j] = net_outputs.lane_lines.mean.left_near[j].y;
            left_near_z[j] = net_outputs.lane_lines.mean.left_near[j].z;
            right_near_y[j] = net_outputs.lane_lines.mean.right_near[j].y;
            right_near_z[j] = net_outputs.lane_lines.mean.right_near[j].z;
            right_far_y[j] = net_outputs.lane_lines.mean.right_far[j].y;
            right_far_z[j] = net_outputs.lane_lines.mean.right_far[j].z;
        }

        data["lane_lines_t"] = plan_t;
        data["lane_lines_x"] = array_to_vector(X_IDXS_FLOAT);
        data["lane_lines_y0"] = left_far_y;
        data["lane_lines_z0"] = left_far_z;
        data["lane_lines_y1"] = left_near_y;
        data["lane_lines_z1"] = left_near_z;
        data["lane_lines_y2"] = right_near_y;
        data["lane_lines_z2"] = right_near_z;
        data["lane_lines_y3"] = right_far_y;
        data["lane_lines_z3"] = right_far_z;

        data["lane_line_stds"] = {
                std::exp(net_outputs.lane_lines.std.left_far[0].y),
                std::exp(net_outputs.lane_lines.std.left_near[0].y),
                std::exp(net_outputs.lane_lines.std.right_near[0].y),
                std::exp(net_outputs.lane_lines.std.right_far[0].y),
        };

        data["lane_line_probs"] = {
                sigmoid(net_outputs.lane_lines.prob.left_far.val),
                sigmoid(net_outputs.lane_lines.prob.left_near.val),
                sigmoid(net_outputs.lane_lines.prob.right_near.val),
                sigmoid(net_outputs.lane_lines.prob.right_far.val),
        };
    }

    // Fill road edges
    {
        std::vector<float> left_y(TRAJECTORY_SIZE), left_z(TRAJECTORY_SIZE);
        std::vector<float> right_y(TRAJECTORY_SIZE), right_z(TRAJECTORY_SIZE);
        for (int j = 0; j < TRAJECTORY_SIZE; j++) {
            left_y[j] = net_outputs.road_edges.mean.left[j].y;
            left_z[j] = net_outputs.road_edges.mean.left[j].z;
            right_y[j] = net_outputs.road_edges.mean.right[j].y;
            right_z[j] = net_outputs.road_edges.mean.right[j].z;
        }

        data["road_edges_t"] = plan_t;
        data["road_edges_x"] = array_to_vector(X_IDXS_FLOAT);
        data["road_edges_y0"] = left_y;
        data["road_edges_z0"] = left_z;
        data["road_edges_y1"] = right_y;
        data["road_edges_z1"] = right_z;

        data["road_edge_stds"] = {
                std::exp(net_outputs.road_edges.std.left[0].y),
                std::exp(net_outputs.road_edges.std.right[0].y),
        };
    }

    // Fill meta
    {
        std::vector<float> desire_state_softmax(DESIRE_LEN);
        softmax(net_outputs.meta.desire_state_prob.array.data(), desire_state_softmax.data(),
                DESIRE_LEN);

        std::vector<float> desire_pred_softmax(DESIRE_PRED_LEN * DESIRE_LEN);
        for (int i = 0; i < DESIRE_PRED_LEN; i++) {
            softmax(net_outputs.meta.desire_pred_prob[i].array.data(),
                    desire_pred_softmax.data() + (i * DESIRE_LEN), DESIRE_LEN);
        }

        std::vector<float> lat_long_t = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
        std::vector<float> gas_disengage_sigmoid(DISENGAGE_LEN), brake_disengage_sigmoid(
                DISENGAGE_LEN),
                steer_override_sigmoid(DISENGAGE_LEN), brake_3ms2_sigmoid(DISENGAGE_LEN),
                brake_4ms2_sigmoid(DISENGAGE_LEN), brake_5ms2_sigmoid(DISENGAGE_LEN);
        for (int i = 0; i < DISENGAGE_LEN; i++) {
            gas_disengage_sigmoid[i] = sigmoid(net_outputs.meta.disengage_prob[i].gas_disengage);
            brake_disengage_sigmoid[i] = sigmoid(
                    net_outputs.meta.disengage_prob[i].brake_disengage);
            steer_override_sigmoid[i] = sigmoid(net_outputs.meta.disengage_prob[i].steer_override);
            brake_3ms2_sigmoid[i] = sigmoid(net_outputs.meta.disengage_prob[i].brake_3ms2);
            brake_4ms2_sigmoid[i] = sigmoid(net_outputs.meta.disengage_prob[i].brake_4ms2);
            brake_5ms2_sigmoid[i] = sigmoid(net_outputs.meta.disengage_prob[i].brake_5ms2);
        }

        std::memmove(prev_brake_5ms2_probs.data(), &prev_brake_5ms2_probs[1], 4 * sizeof(float));
        std::memmove(prev_brake_3ms2_probs.data(), &prev_brake_3ms2_probs[1], 2 * sizeof(float));
        prev_brake_5ms2_probs[4] = brake_5ms2_sigmoid[0];
        prev_brake_3ms2_probs[2] = brake_3ms2_sigmoid[0];

        bool above_fcw_threshold = true;
        for (int i = 0; i < prev_brake_5ms2_probs.size(); i++) {
            float threshold = i < 2 ? FCW_THRESHOLD_5MS2_LOW : FCW_THRESHOLD_5MS2_HIGH;
            above_fcw_threshold = above_fcw_threshold && prev_brake_5ms2_probs[i] > threshold;
        }
        for (int i = 0; i < prev_brake_3ms2_probs.size(); i++) {
            above_fcw_threshold =
                    above_fcw_threshold && prev_brake_3ms2_probs[i] > FCW_THRESHOLD_3MS2;
        }

        data["disengage_t"] = lat_long_t;
        data["gas_disengage_probs"] = gas_disengage_sigmoid;
        data["brake_disengage_probs"] = brake_disengage_sigmoid;
        data["steer_override_probs"] = steer_override_sigmoid;
        data["brake_3ms2_probs"] = brake_3ms2_sigmoid;
        data["brake_4ms2_probs"] = brake_4ms2_sigmoid;
        data["brake_5ms2_probs"] = brake_5ms2_sigmoid;

        data["engaged_prob"] = {sigmoid(net_outputs.meta.engaged_prob)};
        data["desire_prediction"] = desire_pred_softmax;
        data["desire_state"] = desire_state_softmax;
        data["hard_brake_predicted"] = {above_fcw_threshold ? 1.0f : 0.0f};
    }

    // Fill confidence
    {
        const auto &dbps = data.at("brake_disengage_probs");
        const auto &dgps = data.at("gas_disengage_probs");
        const auto &dsps = data.at("steer_override_probs");

        std::vector<float> any_dp(DISENGAGE_LEN);
        std::vector<float> dp_ind(DISENGAGE_LEN);

        for (int i = 0; i < DISENGAGE_LEN; i++) {
            any_dp[i] = 1.0f - ((1.0f - dbps[i]) * (1.0f - dgps[i]) * (1.0f - dsps[i]));
        }

        dp_ind[0] = any_dp[0];
        for (int i = 0; i < DISENGAGE_LEN - 1; i++) {
            dp_ind[i + 1] = (any_dp[i + 1] - any_dp[i]) / (1.0f - any_dp[i]);
        }

        std::memmove(&disengage_buffer[0], &disengage_buffer[DISENGAGE_LEN],
                     sizeof(float) * DISENGAGE_LEN * (DISENGAGE_LEN - 1));
        std::memcpy(&disengage_buffer[DISENGAGE_LEN * (DISENGAGE_LEN - 1)], dp_ind.data(),
                    sizeof(float) * DISENGAGE_LEN);

        float score = 0.0f;
        for (int i = 0; i < DISENGAGE_LEN; i++) {
            score += disengage_buffer[i * DISENGAGE_LEN + DISENGAGE_LEN - 1 - i] / DISENGAGE_LEN;
        }

        if (score < RYG_GREEN) {
            confidence = ConfidenceClass::GREEN;
        } else if (score < RYG_YELLOW) {
            confidence = ConfidenceClass::YELLOW;
        } else {
            confidence = ConfidenceClass::RED;
        }
    }

    // Fill lead
    {
        std::vector<float> t_offsets = {0.0f, 2.0f, 4.0f};
        std::vector<float> leads_prob, leads_prob_time = t_offsets;
        std::vector<float> lead_t = {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f};
        data["leads_t"] = lead_t;
        for (int i = 0; i < LEAD_MHP_SELECTION; i++) {
            leads_prob.push_back(sigmoid(net_outputs.leads.prob[i]));
            const auto &best_prediction = net_outputs.leads.get_best_prediction(i);
            std::vector<float> lead_x(LEAD_TRAJ_LEN), lead_y(LEAD_TRAJ_LEN), lead_v(
                    LEAD_TRAJ_LEN), lead_a(LEAD_TRAJ_LEN);
            std::vector<float> lead_x_std(LEAD_TRAJ_LEN), lead_y_std(LEAD_TRAJ_LEN), lead_v_std(
                    LEAD_TRAJ_LEN), lead_a_std(LEAD_TRAJ_LEN);
            for (int j = 0; j < LEAD_TRAJ_LEN; j++) {
                lead_x[j] = best_prediction.mean[j].x;
                lead_y[j] = best_prediction.mean[j].y;
                lead_v[j] = best_prediction.mean[j].velocity;
                lead_a[j] = best_prediction.mean[j].acceleration;
                lead_x_std[j] = std::exp(best_prediction.std[j].x);
                lead_y_std[j] = std::exp(best_prediction.std[j].y);
                lead_v_std[j] = std::exp(best_prediction.std[j].velocity);
                lead_a_std[j] = std::exp(best_prediction.std[j].acceleration);
            }
            data["leads_x" + std::to_string(i)] = lead_x;
            data["leads_y" + std::to_string(i)] = lead_y;
            data["leads_v" + std::to_string(i)] = lead_v;
            data["leads_a" + std::to_string(i)] = lead_a;
            data["leads_x_std" + std::to_string(i)] = lead_x_std;
            data["leads_y_std" + std::to_string(i)] = lead_y_std;
            data["leads_v_std" + std::to_string(i)] = lead_v_std;
            data["leads_a_std" + std::to_string(i)] = lead_a_std;
        }
        data["leads_prob"] = leads_prob;
        data["leads_prob_time"] = leads_prob_time;
    }

    // Temporal pose
    const auto &v_mean = net_outputs.temporal_pose.velocity_mean;
    const auto &r_mean = net_outputs.temporal_pose.rotation_mean;
    const auto &v_std = net_outputs.temporal_pose.velocity_std;
    const auto &r_std = net_outputs.temporal_pose.rotation_std;
    data["temporal_trans"] = {v_mean.x, v_mean.y, v_mean.z};
    data["temporal_rot"] = {r_mean.x, r_mean.y, r_mean.z};
    data["temporal_trans_std"] = {std::exp(v_std.x), std::exp(v_std.y), std::exp(v_std.z)};
    data["temporal_rot_std"] = {std::exp(r_std.x), std::exp(r_std.y), std::exp(r_std.z)};
}


map<pair<cl_kernel, int>, string> g_args;
map<pair<cl_kernel, int>, int> g_args_size;
map<cl_program, string> g_program_source;

Thneed *g_thneed = NULL;
int g_fd = -1;

// Define function pointer types
typedef cl_program (*clCreateProgramWithSource_t)(cl_context, cl_uint, const char **,
                                                  const size_t *, cl_int *);

typedef cl_int (*clBuildProgram_t)(cl_program, cl_uint, const cl_device_id *, const char *,
                                   void (*pfn_notify)(cl_program, void *), void *);

typedef cl_program (*clCreateProgramWithBinary_t)(cl_context, cl_uint, const cl_device_id *,
                                                  const size_t *, const unsigned char **, cl_int *,
                                                  cl_int *);

typedef cl_int (*clGetPlatformIDs_t)(cl_uint, cl_platform_id *, cl_uint *);

typedef cl_int (*clGetDeviceIDs_t)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *,
                                   cl_uint *);

typedef cl_mem (*clCreateBuffer_t)(cl_context, cl_mem_flags, size_t, void *, cl_int *);

typedef cl_mem (*clCreateImage_t)(cl_context, cl_mem_flags, const cl_image_format *,
                                  const cl_image_desc *, void *, cl_int *);

typedef void *(*clEnqueueMapBuffer_t)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t,
                                      size_t, cl_uint, const cl_event *, cl_event *, cl_int *);

typedef cl_int (*clFinish_t)(cl_command_queue);

typedef cl_int (*clGetMemObjectInfo_t)(cl_mem, cl_mem_info, size_t, void *, size_t *);

typedef cl_int (*clGetImageInfo_t)(cl_mem, cl_image_info, size_t, void *, size_t *);

typedef cl_context (*clCreateContext_t)(const cl_context_properties *, cl_uint,
                                        const cl_device_id *,
                                        void (CL_CALLBACK *)(const char *, const void *, size_t,
                                                             void *), void *, cl_int *);

typedef cl_command_queue (*clCreateCommandQueueWithProperties_t)(cl_context, cl_device_id,
                                                                 const cl_queue_properties *,
                                                                 cl_int *);

typedef cl_int (*clEnqueueWriteBuffer_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                         const void *, cl_uint, const cl_event *, cl_event *);

typedef cl_int (*clEnqueueReadBuffer_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *,
                                        cl_uint, const cl_event *, cl_event *);

typedef cl_int (*clReleaseMemObject_t)(cl_mem);

// Load the OpenCL library
void *opencl_library = dlopen("libOpenCL.so", RTLD_LAZY | RTLD_LOCAL);

// Get function pointers
auto p_clCreateProgramWithSource = reinterpret_cast<clCreateProgramWithSource_t>(dlsym(
        opencl_library, "clCreateProgramWithSource"));
auto p_clBuildProgram = reinterpret_cast<clBuildProgram_t>(dlsym(opencl_library, "clBuildProgram"));
auto p_clCreateProgramWithBinary = reinterpret_cast<clCreateProgramWithBinary_t>(dlsym(
        opencl_library, "clCreateProgramWithBinary"));
auto p_clGetPlatformIDs = reinterpret_cast<clGetPlatformIDs_t>(dlsym(opencl_library,
                                                                     "clGetPlatformIDs"));
auto p_clGetDeviceIDs = reinterpret_cast<clGetDeviceIDs_t>(dlsym(opencl_library, "clGetDeviceIDs"));
auto p_clCreateBuffer = reinterpret_cast<clCreateBuffer_t>(dlsym(opencl_library, "clCreateBuffer"));
auto p_clCreateImage = reinterpret_cast<clCreateImage_t>(dlsym(opencl_library, "clCreateImage"));
auto p_clEnqueueMapBuffer = reinterpret_cast<clEnqueueMapBuffer_t>(dlsym(opencl_library,
                                                                         "clEnqueueMapBuffer"));
auto p_clFinish = reinterpret_cast<clFinish_t>(dlsym(opencl_library, "clFinish"));
auto p_clGetMemObjectInfo = reinterpret_cast<clGetMemObjectInfo_t>(dlsym(opencl_library,
                                                                         "clGetMemObjectInfo"));
auto p_clGetImageInfo = reinterpret_cast<clGetImageInfo_t>(dlsym(opencl_library, "clGetImageInfo"));
auto p_clCreateContext = reinterpret_cast<clCreateContext_t>(dlsym(opencl_library,
                                                                   "clCreateContext"));
auto p_clCreateCommandQueueWithProperties = reinterpret_cast<clCreateCommandQueueWithProperties_t>(dlsym(
        opencl_library, "clCreateCommandQueueWithProperties"));
auto p_clEnqueueWriteBuffer = reinterpret_cast<clEnqueueWriteBuffer_t>(dlsym(opencl_library,
                                                                             "clEnqueueWriteBuffer"));
auto p_clEnqueueReadBuffer = reinterpret_cast<clEnqueueReadBuffer_t>(dlsym(opencl_library,
                                                                           "clEnqueueReadBuffer"));
auto p_clReleaseMemObject = reinterpret_cast<clReleaseMemObject_t>(dlsym(opencl_library,
                                                                         "clReleaseMemObject"));

// Define more function pointer types
typedef cl_kernel (*clCreateKernel_t)(cl_program, const char *, cl_int *);

typedef cl_int (*clGetKernelArgInfo_t)(cl_kernel, cl_uint, cl_kernel_arg_info, size_t, void *,
                                       size_t *);

typedef cl_int (*clEnqueueNDRangeKernel_t)(cl_command_queue, cl_kernel, cl_uint, const size_t *,
                                           const size_t *, const size_t *, cl_uint,
                                           const cl_event *, cl_event *);

typedef cl_int (*clGetKernelInfo_t)(cl_kernel, cl_kernel_info, size_t, void *, size_t *);

typedef cl_int (*clSetKernelArg_t)(cl_kernel, cl_uint, size_t, const void *);

// Get more function pointers
auto p_clCreateKernel = reinterpret_cast<clCreateKernel_t>(dlsym(opencl_library, "clCreateKernel"));
auto p_clGetKernelArgInfo = reinterpret_cast<clGetKernelArgInfo_t>(dlsym(opencl_library,
                                                                         "clGetKernelArgInfo"));
auto p_clEnqueueNDRangeKernel = reinterpret_cast<clEnqueueNDRangeKernel_t>(dlsym(opencl_library,
                                                                                 "clEnqueueNDRangeKernel"));
auto p_clGetKernelInfo = reinterpret_cast<clGetKernelInfo_t>(dlsym(opencl_library,
                                                                   "clGetKernelInfo"));
auto p_clSetKernelArg = reinterpret_cast<clSetKernelArg_t>(dlsym(opencl_library, "clSetKernelArg"));

// Now you can use these function pointers as if they were the original functions
// For example:
// cl_kernel kernel = (*p_clCreateKernel)(program, "my_kernel", &err);


// Now you can use these function pointers as if they were the original functions
// For example:
// cl_context context = (*p_clCreateContext)(NULL, 1, &device_id, NULL, NULL, &err);

#undef assert
#define assert(x) ((x) ? __assert_no_op : (void)__android_log_print(ANDROID_LOG_ERROR, "ASSERT", "Assert failed: %s", #x))

void hexdump(uint8_t *d, int len) {
    assert((len % 4) == 0);
    __android_log_print(ANDROID_LOG_INFO, "JNILOG", "  dumping %p len 0x%x\n", d, len);
    for (int i = 0; i < len / 4; i++) {
        if (i != 0 && (i % 0x10) == 0) __android_log_print(ANDROID_LOG_INFO, "JNILOG", "\n");
        __android_log_print(ANDROID_LOG_INFO, "JNILOG", "%8x ", d[i]);
    }
    __android_log_print(ANDROID_LOG_INFO, "JNILOG", "\n");
}

extern map<cl_program, string> g_program_source;

template<typename Func, typename Id, typename Name>
std::string get_info(Func get_info_func, Id id, Name param_name) {
    size_t size = 0;
    CL_CHECK(get_info_func(id, param_name, 0, NULL, &size));
    std::string info(size, '\0');
    CL_CHECK(get_info_func(id, param_name, size, info.data(), NULL));
    return info;
}

inline std::string get_platform_info(cl_platform_id id, cl_platform_info name) {
    return get_info(&clGetPlatformInfo, id, name);
}

cl_program cl_program_from_source(cl_context ctx, cl_device_id device_id, const std::string &src,
                                  const char *args) {
    const char *csrc = src.c_str();
    cl_program prg = CL_CHECK_ERR((*p_clCreateProgramWithSource)(ctx, 1, &csrc, NULL, &err));
    if (int err = (*p_clBuildProgram)(prg, 1, &device_id, args, NULL, NULL); err != 0) {
        assert(0);
    }
    return prg;
}

cl_program
cl_program_from_binary(cl_context ctx, cl_device_id device_id, const uint8_t *binary, size_t length,
                       const char *args) {
    cl_program prg = CL_CHECK_ERR(
            (*p_clCreateProgramWithBinary)(ctx, 1, &device_id, &length, &binary, NULL, &err));
    if (int err = (*p_clBuildProgram)(prg, 1, &device_id, args, NULL, NULL); err != 0) {
        assert(0);
    }
    return prg;
}

cl_device_id cl_get_device_id(cl_device_type device_type) {
    cl_uint num_platforms = 0;
    CL_CHECK((*p_clGetPlatformIDs)(0, NULL, &num_platforms));
    std::unique_ptr<cl_platform_id[]> platform_ids = std::make_unique<cl_platform_id[]>(
            num_platforms);
    CL_CHECK((*p_clGetPlatformIDs)(num_platforms, &platform_ids[0], NULL));

    for (size_t i = 0; i < num_platforms; ++i) {
        // Get first device
        if (cl_device_id device_id = NULL;
                (*p_clGetDeviceIDs)(platform_ids[i], device_type, 1, &device_id, NULL) == 0 &&
                device_id) {
            return device_id;
        }
    }
    assert(0);
    return nullptr;
}

#include <dirent.h>
#include <unistd.h>
#include <iostream>

cl_int
thneed_clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void *arg_value) {
    g_args_size[make_pair(kernel, arg_index)] = arg_size;
    if (arg_value != NULL) {
        g_args[make_pair(kernel, arg_index)] = string((char *) arg_value, arg_size);
    } else {
        g_args[make_pair(kernel, arg_index)] = string("");
    }
    cl_int ret = (*p_clSetKernelArg)(kernel, arg_index, arg_size, arg_value);
    return ret;
}

void getGPUMemoryAllocationFD() {
    DIR *dir = opendir("/proc/self/fd");
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string fdPath = std::string("/proc/self/fd/") + entry->d_name;
        char linkTarget[256];
        ssize_t len = readlink(fdPath.c_str(), linkTarget, sizeof(linkTarget) - 1);
        if (len != -1) {
            linkTarget[len] = '\0';
            if (std::string(linkTarget) == "/dev/kgsl-3d0") {
                g_fd = std::stoi(entry->d_name);
                __android_log_print(ANDROID_LOG_INFO, "JNILOG",
                                    "File descriptor found for GPU allocation: %d", g_fd);
                closedir(dir);
                return;
            }
        }
    }

    // hmm, didn't find anything...
    closedir(dir);
}

std::string readFileIntoString(const char *filepath) {
    std::ifstream ifs(filepath);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
}

void Thneed::load(const char *filename) {
    __android_log_print(ANDROID_LOG_INFO, "JNILOG", "Thneed::load: loading from %s\n", filename);

    string buf = readFileIntoString(filename);
    int jsz = *(int *) buf.data();
    string jsonerr;
    string jj(buf.data() + sizeof(int), jsz);
    json11::Json jdat = json11::Json::parse(jj, jsonerr);

    map<cl_mem, cl_mem> real_mem;
    real_mem[NULL] = NULL;

    int ptr = sizeof(int) + jsz;
    for (auto &obj: jdat["objects"].array_items()) {
        auto mobj = obj.object_items();
        int sz = mobj["size"].int_value();
        cl_mem clbuf = NULL;

        if (mobj["buffer_id"].string_value().size() > 0) {
            // image buffer must already be allocated
            clbuf = real_mem[*(cl_mem *) (mobj["buffer_id"].string_value().data())];
            assert(mobj["needs_load"].bool_value() == false);
        } else {
            if (mobj["needs_load"].bool_value()) {
                clbuf = (*p_clCreateBuffer)(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, sz,
                                            &buf[ptr], NULL);
                if (debug >= 1)
                    __android_log_print(ANDROID_LOG_INFO, "JNILOG", "loading %p %d @ 0x%X\n", clbuf,
                                        sz, ptr);
                ptr += sz;
            } else {
                // TODO: is there a faster way to init zeroed out buffers?
                void *host_zeros = calloc(sz, 1);
                clbuf = (*p_clCreateBuffer)(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, sz,
                                            host_zeros, NULL);
                free(host_zeros);
            }
        }
        assert(clbuf != NULL);

        if (mobj["arg_type"] == "image2d_t" || mobj["arg_type"] == "image1d_t") {
            cl_image_desc desc = {0};
            desc.image_type = (mobj["arg_type"] == "image2d_t") ? CL_MEM_OBJECT_IMAGE2D
                                                                : CL_MEM_OBJECT_IMAGE1D_BUFFER;
            desc.image_width = mobj["width"].int_value();
            desc.image_height = mobj["height"].int_value();
            desc.image_row_pitch = mobj["row_pitch"].int_value();
            assert(sz == desc.image_height * desc.image_row_pitch);
            desc.buffer = clbuf;
            cl_image_format format = {0};
            format.image_channel_order = CL_RGBA;
            format.image_channel_data_type = mobj["float32"].bool_value() ? CL_FLOAT
                                                                          : CL_HALF_FLOAT;

            cl_int errcode;

            clbuf = (*p_clCreateImage)(context, CL_MEM_READ_WRITE, &format, &desc, NULL, &errcode);

            if (clbuf == NULL) {
                __android_log_print(ANDROID_LOG_INFO, "JNILOG",
                                    "clError: %d create image %zux%zu rp %zu with buffer %p\n",
                                    errcode,
                                    desc.image_width, desc.image_height, desc.image_row_pitch,
                                    desc.buffer);
            }
            assert(clbuf != NULL);
        }

        real_mem[*(cl_mem *) (mobj["id"].string_value().data())] = clbuf;
    }

    map<string, cl_program> g_programs;
    for (const auto &[name, source]: jdat["programs"].object_items()) {
        if (debug >= 1)
            __android_log_print(ANDROID_LOG_INFO, "JNILOG", "building %s with size %zu\n",
                                name.c_str(), source.string_value().size());
        g_programs[name] = cl_program_from_source(context, device_id, source.string_value());
    }

    for (auto &obj: jdat["inputs"].array_items()) {
        auto mobj = obj.object_items();
        int sz = mobj["size"].int_value();
        cl_mem aa = real_mem[*(cl_mem *) (mobj["buffer_id"].string_value().data())];
        input_clmem.push_back(aa);
        input_sizes.push_back(sz);
        __android_log_print(ANDROID_LOG_INFO, "JNILOG",
                            "Thneed::load: adding input %s with size %d\n",
                            mobj["name"].string_value().data(), sz);

        cl_int cl_err;
        void *ret = (*p_clEnqueueMapBuffer)(command_queue, aa, CL_TRUE, CL_MAP_WRITE, 0, sz, 0,
                                            NULL, NULL, &cl_err);
        //if (cl_err != CL_SUCCESS) __android_log_print(ANDROID_LOG_INFO, "JNILOG","clError: %s map %p %d\n", cl_get_error_string(cl_err), aa, sz);
        assert(cl_err == CL_SUCCESS);
        inputs.push_back(ret);
    }

    for (auto &obj: jdat["outputs"].array_items()) {
        auto mobj = obj.object_items();
        int sz = mobj["size"].int_value();
        __android_log_print(ANDROID_LOG_INFO, "JNILOG",
                            "Thneed::save: adding output with size %d\n", sz);
        // TODO: support multiple outputs
        output = real_mem[*(cl_mem *) (mobj["buffer_id"].string_value().data())];
        if (output == NULL)
            __android_log_print(ANDROID_LOG_INFO, "JNILOG", "Thneed::save: output was null!");
    }

    for (auto &obj: jdat["binaries"].array_items()) {
        string name = obj["name"].string_value();
        size_t length = obj["length"].int_value();
        if (debug >= 1)
            __android_log_print(ANDROID_LOG_INFO, "JNILOG", "binary %s with size %zu\n",
                                name.c_str(), length);
        g_programs[name] = cl_program_from_binary(context, device_id, (const uint8_t *) &buf[ptr],
                                                  length);
        ptr += length;
    }

    for (auto &obj: jdat["kernels"].array_items()) {
        auto gws = obj["global_work_size"];
        auto lws = obj["local_work_size"];
        auto kk = shared_ptr<CLQueuedKernel>(new CLQueuedKernel(this));

        kk->name = obj["name"].string_value();
        kk->program = g_programs[kk->name];
        kk->work_dim = obj["work_dim"].int_value();
        for (int i = 0; i < kk->work_dim; i++) {
            kk->global_work_size[i] = gws[i].int_value();
            kk->local_work_size[i] = lws[i].int_value();
        }
        kk->num_args = obj["num_args"].int_value();
        for (int i = 0; i < kk->num_args; i++) {
            string arg = obj["args"].array_items()[i].string_value();
            int arg_size = obj["args_size"].array_items()[i].int_value();
            kk->args_size.push_back(arg_size);
            if (arg_size == 8) {
                cl_mem val = *(cl_mem *) (arg.data());
                val = real_mem[val];
                kk->args.push_back(string((char *) &val, sizeof(val)));
            } else {
                kk->args.push_back(arg);
            }
        }
        kq.push_back(kk);
    }

    (*p_clFinish)(command_queue);
}

// *********** Thneed ***********

#ifndef QCOM2

Thneed::Thneed(bool do_clinit, cl_context _context) {
    context = _context;
    if (do_clinit) clinit();
    debug = 0; //(thneed_debug_env != NULL) ? atoi(thneed_debug_env) : 0;
}

void Thneed::execute(float **finputs, float *foutput, bool slow) {
    uint64_t tb, te;
    if (debug >= 1) tb = nanos_since_boot();

    // ****** copy inputs
    copy_inputs(finputs);

    // ****** run commands
    clexec();

    // ****** copy outputs
    copy_output(foutput);

    if (debug >= 1) {
        te = nanos_since_boot();
        __android_log_print(ANDROID_LOG_INFO, "JNILOG", "model exec in %lu us\n", (te - tb) / 1000);
    }
}

#else

#endif

void Thneed::stop() {
    //__android_log_print(ANDROID_LOG_INFO, "JNILOG","Thneed::stop: recorded %lu commands\n", cmds.size());
    record = false;
}

void Thneed::clinit() {
    device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
    if (context == NULL)
        context = CL_CHECK_ERR((*p_clCreateContext)(NULL, 1, &device_id, NULL, NULL, &err));
    //cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue_properties props[3] = {CL_QUEUE_PROPERTIES, 0, 0};
    command_queue = CL_CHECK_ERR(
            (*p_clCreateCommandQueueWithProperties)(context, device_id, props, &err));
    __android_log_print(ANDROID_LOG_INFO, "JNILOG", "Thneed::clinit done\n");
}

cl_int Thneed::clexec() {
    if (debug >= 1)
        __android_log_print(ANDROID_LOG_INFO, "JNILOG",
                            "Thneed::clexec: running %lu queued kernels\n", kq.size());
    for (auto &k: kq) {
        if (record) ckq.push_back(k);
        cl_int ret = k->exec();
        assert(ret == CL_SUCCESS);
    }
    return (*p_clFinish)(command_queue);
}

void Thneed::copy_inputs(float **finputs, bool internal) {
    for (int idx = 0; idx < inputs.size(); ++idx) {
        if (debug >= 1)
            __android_log_print(ANDROID_LOG_INFO, "JNILOG",
                                "copying idx:%d %lu -- %p -> %p (cl %p)\n", idx, input_sizes[idx],
                                finputs[idx], inputs[idx], input_clmem[idx]);

        if (internal) {
            // if it's internal, using memcpy is fine since the buffer sync is cached in the ioctl layer
            if (finputs[idx] != NULL) memcpy(inputs[idx], finputs[idx], input_sizes[idx]);
        } else {
            if (finputs[idx] != NULL)
                CL_CHECK((*p_clEnqueueWriteBuffer)(command_queue, input_clmem[idx], CL_TRUE, 0,
                                                   input_sizes[idx], finputs[idx], 0, NULL, NULL));
        }
    }
}

void Thneed::copy_output(float *foutput) {
    if (output != NULL) {
        size_t sz;
        (*p_clGetMemObjectInfo)(output, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
        if (debug >= 1)
            __android_log_print(ANDROID_LOG_INFO, "JNILOG", "copying %lu for output %p -> %p\n", sz,
                                output, foutput);
        CL_CHECK((*p_clEnqueueReadBuffer)(command_queue, output, CL_TRUE, 0, sz, foutput, 0, NULL,
                                          NULL));
    } else {
        __android_log_print(ANDROID_LOG_INFO, "JNILOG",
                            "CAUTION: model output is NULL, does it have no outputs?\n");
    }
}

// *********** CLQueuedKernel ***********

CLQueuedKernel::CLQueuedKernel(Thneed *lthneed,
                               cl_kernel _kernel,
                               cl_uint _work_dim,
                               const size_t *_global_work_size,
                               const size_t *_local_work_size) {
    thneed = lthneed;
    kernel = _kernel;
    work_dim = _work_dim;
    assert(work_dim <= 3);
    for (int i = 0; i < work_dim; i++) {
        global_work_size[i] = _global_work_size[i];
        local_work_size[i] = _local_work_size[i];
    }

    char _name[0x100];
    (*p_clGetKernelInfo)(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(_name), _name, NULL);
    name = string(_name);
    (*p_clGetKernelInfo)(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args), &num_args, NULL);

    // get args
    for (int i = 0; i < num_args; i++) {
        char arg_name[0x100] = {0};
        (*p_clGetKernelArgInfo)(kernel, i, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name, NULL);
        arg_names.push_back(string(arg_name));
        (*p_clGetKernelArgInfo)(kernel, i, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_name), arg_name,
                                NULL);
        arg_types.push_back(string(arg_name));

        args.push_back(g_args[make_pair(kernel, i)]);
        args_size.push_back(g_args_size[make_pair(kernel, i)]);
    }

    // get program
    (*p_clGetKernelInfo)(kernel, CL_KERNEL_PROGRAM, sizeof(program), &program, NULL);
}

int CLQueuedKernel::get_arg_num(const char *search_arg_name) {
    for (int i = 0; i < num_args; i++) {
        if (arg_names[i] == search_arg_name) return i;
    }
    __android_log_print(ANDROID_LOG_INFO, "JNILOG", "failed to find %s in %s\n", search_arg_name,
                        name.c_str());
    assert(false);
}

cl_int CLQueuedKernel::exec() {
    if (kernel == NULL) {
        kernel = (*p_clCreateKernel)(program, name.c_str(), NULL);
        arg_names.clear();
        arg_types.clear();

        for (int j = 0; j < num_args; j++) {
            char arg_name[0x100] = {0};
            (*p_clGetKernelArgInfo)(kernel, j, CL_KERNEL_ARG_NAME, sizeof(arg_name), arg_name,
                                    NULL);
            arg_names.push_back(string(arg_name));
            (*p_clGetKernelArgInfo)(kernel, j, CL_KERNEL_ARG_TYPE_NAME, sizeof(arg_name), arg_name,
                                    NULL);
            arg_types.push_back(string(arg_name));

            cl_int ret;
            if (args[j].size() != 0) {
                assert(args[j].size() == args_size[j]);
                ret = thneed_clSetKernelArg(kernel, j, args[j].size(), args[j].data());
            } else {
                ret = thneed_clSetKernelArg(kernel, j, args_size[j], NULL);
            }
            assert(ret == CL_SUCCESS);
        }
    }

    //if (thneed->debug >= 1) {
    //    debug_print(thneed->debug >= 2);
    //}

    return (*p_clEnqueueNDRangeKernel)(thneed->command_queue,
                                       kernel, work_dim, NULL, global_work_size, local_work_size, 0,
                                       NULL, NULL);
}

void CLQueuedKernel::debug_print(bool verbose) {
    __android_log_print(ANDROID_LOG_INFO, "JNILOG", "%p %56s -- ", kernel, name.c_str());
    for (int i = 0; i < work_dim; i++) {
        __android_log_print(ANDROID_LOG_INFO, "JNILOG", "%4zu ", global_work_size[i]);
    }
    __android_log_print(ANDROID_LOG_INFO, "JNILOG", " -- ");
    for (int i = 0; i < work_dim; i++) {
        __android_log_print(ANDROID_LOG_INFO, "JNILOG", "%4zu ", local_work_size[i]);
    }
    __android_log_print(ANDROID_LOG_INFO, "JNILOG", "\n");

    if (verbose) {
        for (int i = 0; i < num_args; i++) {
            string arg = args[i];
            __android_log_print(ANDROID_LOG_INFO, "JNILOG", "  %s %s", arg_types[i].c_str(),
                                arg_names[i].c_str());
            void *arg_value = (void *) arg.data();
            int arg_size = arg.size();
            if (arg_size == 0) {
                __android_log_print(ANDROID_LOG_INFO, "JNILOG", " (size) %d", args_size[i]);
            } else if (arg_size == 1) {
                __android_log_print(ANDROID_LOG_INFO, "JNILOG", " = %d", *((char *) arg_value));
            } else if (arg_size == 2) {
                __android_log_print(ANDROID_LOG_INFO, "JNILOG", " = %d", *((short *) arg_value));
            } else if (arg_size == 4) {
                if (arg_types[i] == "float") {
                    __android_log_print(ANDROID_LOG_INFO, "JNILOG", " = %f",
                                        *((float *) arg_value));
                } else {
                    __android_log_print(ANDROID_LOG_INFO, "JNILOG", " = %d", *((int *) arg_value));
                }
            } else if (arg_size == 8) {
                cl_mem val = (cl_mem) (*((uintptr_t *) arg_value));
                __android_log_print(ANDROID_LOG_INFO, "JNILOG", " = %p", val);
                if (val != NULL) {
                    cl_mem_object_type obj_type;
                    (*p_clGetMemObjectInfo)(val, CL_MEM_TYPE, sizeof(obj_type), &obj_type, NULL);
                    if (arg_types[i] == "image2d_t" || arg_types[i] == "image1d_t" ||
                        obj_type == CL_MEM_OBJECT_IMAGE2D) {
                        cl_image_format format;
                        size_t width, height, depth, array_size, row_pitch, slice_pitch;
                        cl_mem buf;
                        (*p_clGetImageInfo)(val, CL_IMAGE_FORMAT, sizeof(format), &format, NULL);
                        assert(format.image_channel_order == CL_RGBA);
                        assert(format.image_channel_data_type == CL_HALF_FLOAT ||
                               format.image_channel_data_type == CL_FLOAT);
                        (*p_clGetImageInfo)(val, CL_IMAGE_WIDTH, sizeof(width), &width, NULL);
                        (*p_clGetImageInfo)(val, CL_IMAGE_HEIGHT, sizeof(height), &height, NULL);
                        (*p_clGetImageInfo)(val, CL_IMAGE_ROW_PITCH, sizeof(row_pitch), &row_pitch,
                                            NULL);
                        (*p_clGetImageInfo)(val, CL_IMAGE_DEPTH, sizeof(depth), &depth, NULL);
                        (*p_clGetImageInfo)(val, CL_IMAGE_ARRAY_SIZE, sizeof(array_size),
                                            &array_size, NULL);
                        (*p_clGetImageInfo)(val, CL_IMAGE_SLICE_PITCH, sizeof(slice_pitch),
                                            &slice_pitch, NULL);
                        assert(depth == 0);
                        assert(array_size == 0);
                        assert(slice_pitch == 0);

                        (*p_clGetImageInfo)(val, CL_IMAGE_BUFFER, sizeof(buf), &buf, NULL);
                        size_t sz = 0;
                        if (buf != NULL)
                            (*p_clGetMemObjectInfo)(buf, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
                        __android_log_print(ANDROID_LOG_INFO, "JNILOG",
                                            " image %zu x %zu rp %zu @ %p buffer %zu", width,
                                            height, row_pitch, buf, sz);
                    } else {
                        size_t sz;
                        (*p_clGetMemObjectInfo)(val, CL_MEM_SIZE, sizeof(sz), &sz, NULL);
                        __android_log_print(ANDROID_LOG_INFO, "JNILOG", " buffer %zu", sz);
                    }
                }
            }
            __android_log_print(ANDROID_LOG_INFO, "JNILOG", "\n");
        }
    }
}

#endif // USE_PRECOMPILED

ThneedModel::ThneedModel(const std::string path, float *_output, size_t _output_size, int runtime,
                         bool luse_tf8, cl_context context) {
    thneed = new Thneed(true, context);
    thneed->load(path.c_str());
    thneed->clexec();

    recorded = false;
    output = _output;
}

void *ThneedModel::getCLBuffer(const std::string name) {
    int index = -1;
    for (int i = 0; i < inputs.size(); i++) {
        if (name == inputs[i]->name) {
            index = i;
            break;
        }
    }

    if (thneed->input_clmem.size() >= inputs.size()) {
        return &thneed->input_clmem[inputs.size() - index - 1];
    } else {
        return nullptr;
    }
}

void ThneedModel::execute() {
    if (!recorded) {
        thneed->record = true;
        float *input_buffers[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            input_buffers[inputs.size() - i - 1] = inputs[i]->buffer;
        }

        thneed->copy_inputs(input_buffers);
        thneed->clexec();
        thneed->copy_output(output);
        thneed->stop();

        recorded = true;
    } else {
        float *input_buffers[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            input_buffers[inputs.size() - i - 1] = inputs[i]->buffer;
        }
        thneed->execute(input_buffers, output);
    }
}


const int OUTPUT_SIZE = 5992;
const int LATERAL_CONTROL_PARAMS_LEN = 2;
const int PREV_DESIRED_CURVS_LEN = 1 * (HISTORY_BUFFER_LEN + 1);
const int DESIRED_CURV_WIDTH = 1;

std::string *pathString;
jfloat *outputs;
jint output_len;
ThneedModel *thneed;
float *zero_buf;
float *features_buf;
float *prev_curvs_buf;
int zero_len = 1024 / 4;
int features_len = HISTORY_BUFFER_LEN * FEATURE_LEN;

extern "C" {
// JNI function to initialize the array pool
void JNICALL
Java_com_example_theno_CameraManager_initArrayPool(JNIEnv *env, jclass clazz,
                                                   jobjectArray arrayPool) {
    // TODO: implement initArrayPool()
    // Store global references to the preallocated jfloatArray objects
    jsize poolSize = env->GetArrayLength(arrayPool);

    if (poolSize != 84) { // Total number of float arrays in ModelMap
        LOGI("%s", reinterpret_cast<const char *>(poolSize));
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "Array pool size mismatch Total number of float arrays in ModelMap DIFFERENT ");
        return;
    }

    static jfloatArray globalPool[84];
    const char *keys[] = {
            "position_t", "position_x", "position_y", "position_z", "position_x_std",
            "position_y_std", "position_z_std",
            "velocity_t", "velocity_x", "velocity_y", "velocity_z",
            "acceleration_t", "acceleration_x", "acceleration_y", "acceleration_z",
            "orientation_t", "orientation_x", "orientation_y", "orientation_z",
            "orientation_rate_t", "orientation_rate_x", "orientation_rate_y", "orientation_rate_z",
            "lane_lines_t", "lane_lines_x",
            "lane_lines_y0", "lane_lines_y1", "lane_lines_y2", "lane_lines_y3",
            "lane_lines_z0", "lane_lines_z1", "lane_lines_z2", "lane_lines_z3",
            "lane_line_stds", "lane_line_probs",
            "road_edges_t", "road_edges_x", "road_edges_y0", "road_edges_y1",
            "road_edges_z0", "road_edges_z1", "road_edge_stds",
            "leads_t", "leads_prob", "leads_prob_time",
            "leads_x0", "leads_y0", "leads_v0", "leads_a0",
            "leads_x_std0", "leads_y_std0", "leads_v_std0", "leads_a_std0",
            "leads_x1", "leads_y1", "leads_v1", "leads_a1",
            "leads_x_std1", "leads_y_std1", "leads_v_std1", "leads_a_std1",
            "leads_x2", "leads_y2", "leads_v2", "leads_a2",
            "leads_x_std2", "leads_y_std2", "leads_v_std2", "leads_a_std2",
            "disengage_t", "gas_disengage_probs", "brake_disengage_probs", "steer_override_probs",
            "brake_3ms2_probs", "brake_4ms2_probs", "brake_5ms2_probs",
            "engaged_prob", "desire_prediction", "desire_state", "hard_brake_predicted",
            "temporal_trans", "temporal_rot", "temporal_trans_std", "temporal_rot_std"
    };


    for (jsize i = 0; i < poolSize; ++i) {
        jfloatArray array = (jfloatArray) env->GetObjectArrayElement(arrayPool, i);
        globalPool[i] = (jfloatArray) env->NewGlobalRef(array);
        env->DeleteLocalRef(array);
    }
    LOGI("Array pool initialized with %d arrays", poolSize);
}

void JNICALL
Java_com_example_theno_CameraManager_cleanupArrayPool(
        JNIEnv *env,
        jclass /* this */,
        jobjectArray arrayPool) {
    jsize poolSize = env->GetArrayLength(arrayPool);
    for (jsize i = 0; i < poolSize; ++i) {
        jfloatArray array = (jfloatArray) env->GetObjectArrayElement(arrayPool, i);
        env->DeleteGlobalRef(array);
        env->DeleteLocalRef(array);
    }
    LOGI("Array pool cleaned up");
}
/*my loch ends*/


void JNICALL Java_com_example_theno_CameraManager_createStdString(JNIEnv *env, jclass clazz,
                                                                  jstring javaString) {
    // Convert Java string to C++ string
    const char *cString = env->GetStringUTFChars(javaString, 0);
    pathString = new std::string(cString);

    // Release the C string
    env->ReleaseStringUTFChars(javaString, cString);
}

void JNICALL Java_com_example_theno_CameraManager_getArray(JNIEnv *env, jobject obj, jint size) {
    // Allocate a float array of the given size
    outputs = new jfloat[size];
    output_len = size;
    zero_buf = new float[zero_len];
    features_buf = new float[features_len];
    prev_curvs_buf = new float[PREV_DESIRED_CURVS_LEN];
    for (int i = 0; i < zero_len; i++)
        zero_buf[i] = 0;
    for (int i = 0; i < features_len; i++)
        features_buf[i] = 0;
    for (int i = 0; i < PREV_DESIRED_CURVS_LEN; i++)
        prev_curvs_buf[i] = 0;
}

void JNICALL Java_com_example_theno_CameraManager_initThneed(JNIEnv *env, jobject obj) {
    __android_log_print(ANDROID_LOG_INFO, "JNILOG", "muddle path %s", pathString->c_str());

    thneed = new ThneedModel(*pathString, outputs, output_len, 0, false, NULL);

}

JNIEXPORT jint JNICALL Java_com_example_theno_CameraManager_executeModel(JNIEnv *env, jobject obj,
                                                                         jfloatArray input,
                                                                         jobjectArray arrayPool) {
    // buffers
    jfloat *input_buf = env->GetFloatArrayElements(input, 0);

    // useful offsets
    int input_imgs_len = 1572864 / 4;
    int desire_len = 3200 / 4;

    float *input_imgs_buf = &input_buf[0];
    float *big_input_imgs_buf = &input_buf[input_imgs_len];
    float *desire_buf = &input_buf[input_imgs_len * 2];
    float *lat_params = &input_buf[input_imgs_len * 2 + desire_len];

    thneed->setInputBuffer("input_imgs", input_imgs_buf, input_imgs_len);
    thneed->setInputBuffer("big_input_imgs", big_input_imgs_buf, input_imgs_len);
    thneed->setInputBuffer("desire", desire_buf, desire_len);
    thneed->setInputBuffer("traffic_convention", zero_buf, 8 / 4);
    thneed->setInputBuffer("lateral_control_params", lat_params, LATERAL_CONTROL_PARAMS_LEN);
    thneed->setInputBuffer("prev_desired_curvs", prev_curvs_buf, PREV_DESIRED_CURVS_LEN);
    thneed->setInputBuffer("nav_features", zero_buf, 1024 / 4);
    thneed->setInputBuffer("nav_instructions", zero_buf, 600 / 4);
    thneed->setInputBuffer("features_buffer", features_buf, features_len);

    // ok execute model
    thneed->execute();

    // When done, release the memory
    env->ReleaseFloatArrayElements(input, input_buf, 0);

    // handle features
    std::memmove(&features_buf[0], &features_buf[FEATURE_LEN],
                 sizeof(float) * FEATURE_LEN * (HISTORY_BUFFER_LEN - 1));
    std::memcpy(&features_buf[FEATURE_LEN * (HISTORY_BUFFER_LEN - 1)], &outputs[OUTPUT_SIZE],
                sizeof(float) * FEATURE_LEN);

    // handle previous curves
    std::memmove(&prev_curvs_buf[0], &prev_curvs_buf[1], PREV_DESIRED_CURVS_LEN - 1);
    prev_curvs_buf[PREV_DESIRED_CURVS_LEN - 1] = outputs[5990];
    float input_array[] = {-0.5656314492, -0.1568799913, 0.0000506607, -0.1953868866, -0.0109686377, 0.0001600070, -0.1415905654, -0.0043533780, -0.0124333762, 0.0000513002, 0.0003255454, -0.0035311407, 0.0008233793, 0.0045576757, -0.0005346988, -0.5633367896, -0.1567801386, 0.0000437485, -0.2055435181, -0.0126539255, 0.0001536198, -0.1378185600, -0.0053104889, -0.0116390213, 0.0000594650, 0.0003779902, -0.0033784376, 0.0007706563, 0.0043242122, -0.0005186768, -0.5493624210, -0.1574455798, 0.0000160112, -0.2062387466, -0.0134308953, 0.0008815639, -0.1307382435, -0.0072723660, -0.0021298840, 0.0000835889, 0.0005261854, -0.0034483471, 0.0007573787, 0.0036100778, -0.0005295167, -0.5300576687, -0.1572997123, 0.0000274762, -0.2058286667, -0.0121893575, 0.0022447244, -0.1156072840, -0.0060581369, 0.0062277857, 0.0001261322, 0.0006877242, -0.0032968917, 0.0006868166, 0.0024485078, -0.0003868270, -0.5131051540, -0.1580227613, -0.0000322343, -0.2093286514, -0.0115068806, 0.0031606928, -0.1085484698, -0.0064624632, 0.0160180554, 0.0001908476, 0.0007746903, -0.0033659537, 0.0004370369, 0.0009346034, -0.0002071764, -0.5151441097, -0.1578070372, 0.0002268032, -0.2175922394, -0.0121469758, 0.0032734983, -0.1014203802, -0.0062867701, 0.0179059766, 0.0002037889, 0.0007051515, -0.0033345951, 0.0001645362, -0.0000575603, -0.0000208096, -0.5273809433, -0.1636475921, 0.0007167105, -0.2205677032, -0.0118329572, 0.0041237250, -0.0849367455, -0.0036132098, 0.0144092618, 0.0002069432, 0.0006417743, -0.0036869957, -0.0000952090, -0.0000498776, 0.0002382123, -0.5261359215, -0.1684757918, 0.0016105380, -0.2182378769, -0.0100864135, 0.0048319371, -0.0612199120, -0.0001492184, 0.0052334722, 0.0001974398, 0.0007213989, -0.0035318600, -0.0000302565, 0.0004782863, 0.0005339402, -0.5498657227, -0.1735096276, 0.0027453452, -0.2355098724, -0.0102628954, 0.0043802531, -0.0335331261, 0.0055955788, 0.0030107903, 0.0002200778, 0.0008636739, -0.0035696661, 0.0000538611, 0.0005694203, 0.0007719246, -0.5473661423, -0.1758858860, 0.0029014703, -0.2307338715, -0.0075585134, 0.0047673220, -0.0079640299, 0.0086551774, 0.0054967441, 0.0001806731, 0.0009822513, -0.0035032304, 0.0001826805, 0.0002716450, 0.0013720561, -0.5619010925, -0.1779538393, 0.0050322749, -0.2261915207, -0.0055644149, 0.0052481689, 0.0171614885, 0.0111638084, 0.0053735580, 0.0001420545, 0.0009935275, -0.0032816278, 0.0001643876, 0.0001482736, 0.0019869199, -0.5738010406, -0.1867888570, 0.0058737891, -0.1961803436, -0.0035033105, 0.0056722471, 0.0562397838, 0.0147630945, 0.0033269841, 0.0002280429, 0.0010850455, -0.0026006862, 0.0001488996, 0.0000498891, 0.0025811822, -0.5924034119, -0.1862946898, 0.0078289583, -0.1786365509, -0.0020598127, 0.0059945276, 0.0929439217, 0.0152285332, 0.0025957627, 0.0002728994, 0.0010936959, -0.0018619150, 0.0002814753, -0.0000752458, 0.0030965402, -0.6159305573, -0.1945894361, 0.0091866609, -0.1638240814, -0.0025579352, 0.0053351633, 0.1204449832, 0.0143385455, 0.0014038263, 0.0003266188, 0.0010948903, 0.0002828594, 0.0002335752, -0.0001690084, 0.0035397261, -0.6005229950, -0.1929755658, 0.0117614344, -0.1388816833, -0.0023878859, 0.0059146732, 0.1341379583, 0.0145709924, 0.0018974850, 0.0005375081, 0.0010960000, 0.0020251898, 0.0002704462, -0.0000244911, 0.0038622683, -0.6508140564, -0.1996107101, 0.0139323846, -0.1488599777, -0.0031737629, 0.0045721349, 0.1505517066, 0.0078590158, -0.0003196951, 0.0005555317, 0.0011516829, 0.0044645108, 0.0003051401, 0.0002411639, 0.0040263878, -0.6216163635, -0.1949936152, 0.0135067105, -0.1430530548, -0.0059245415, 0.0031573670, 0.1864716709, 0.0033775214, -0.0006290795, 0.0005754576, 0.0014582839, 0.0065646204, 0.0002627257, 0.0004374837, 0.0047982312, -0.6467170715, -0.1878931969, 0.0153577812, -0.0789051056, -0.0080116987, 0.0026909560, 0.2409757972, -0.0001109177, -0.0029579173, 0.0008050965, 0.0018078190, 0.0095275082, 0.0001295834, 0.0006952297, 0.0064798226, -0.5787391663, -0.1881036460, 0.0159500018, -0.0093889236, -0.0106323566, 0.0017874758, 0.3119117916, -0.0023319107, -0.0031032073, 0.0008252310, 0.0021000435, 0.0138142640, -0.0000418173, 0.0009607238, 0.0097597148, -0.5241546631, -0.1893115640, 0.0130488425, 0.1145610809, -0.0028384272, 0.0015819841, 0.4260166883, 0.0213360600, -0.0011996099, 0.0008818188, 0.0027197427, 0.0198167842, -0.0003480494, 0.0012639827, 0.0166879557, -0.4445190430, -0.1911950111, 0.0114582032, 0.2941370010, 0.0102455067, 0.0014435747, 0.5788444877, 0.0571504980, -0.0011060610, 0.0009979592, 0.0034551925, 0.0303324871, -0.0006158549, 0.0017174737, 0.0268115252, -0.2950820923, -0.1838950813, 0.0120176300, 0.6255350113, 0.0342840031, 0.0012452300, 0.7183127403, 0.1150999665, -0.0003026125, 0.0007986509, 0.0044664582, 0.0452284105, -0.0009452800, 0.0019324319, 0.0397577547, -0.0271682739, -0.1474240422, 0.0055268258, 0.9169874191, 0.0620438308, 0.0028378316, 0.8191226721, 0.1726472676, -0.0002724114, 0.0008436840, 0.0057015959, 0.0662057921, -0.0010786263, 0.0021431905, 0.0499778800, 0.5119438171, -0.0346641243, 0.0062284991, 1.3391571045, 0.0750008747, 0.0045458130, 0.8860050440, 0.1740064174, 0.0033187757, 0.0008952506, 0.0072493283, 0.0929338709, -0.0010727344, 0.0017611152, 0.0437436104, 1.2584609985, 0.1623287350, 0.0031202659, 1.8615798950, 0.0523237959, 0.0076963194, 0.9318436384, 0.1437485367, 0.0069420571, 0.0011744436, 0.0082589574, 0.1196109951, -0.0010157831, 0.0012292046, 0.0228935331, 2.2034072876, 0.4307279289, -0.0079239383, 2.4500141144, 0.0019546873, 0.0112193767, 0.9207977057, 0.0920439288, 0.0103974026, 0.0014423574, 0.0091609173, 0.1246655509, -0.0005141939, 0.0003213203, -0.0018762436, 3.4935379028, 0.7392174006, -0.0092754364, 3.0205650330, -0.0506090559, 0.0144434106, 0.9105430841, 0.0171120893, 0.0151127819, 0.0017384333, 0.0095380936, 0.1089678779, 0.0002382711, -0.0008470074, -0.0282836556, 4.8966445923, 0.9593646526, -0.0182263553, 3.5813331604, -0.0986924991, 0.0148157720, 0.9056636691, -0.0812016800, 0.0197805166, 0.0023755743, 0.0090056416, 0.0700209960, 0.0011501790, -0.0018593451, -0.0534359105, 6.7280349731, 1.0718071461, -0.0277942717, 4.1899089813, -0.1477102190, 0.0172397271, 0.9178192019, -0.1992295831, 0.0200599078, 0.0035796841, 0.0077076983, 0.0175186247, 0.0017093611, -0.0023217886, -0.0712060407, 8.7475051880, 1.0385380983, -0.0225174427, 4.7818665504, -0.1815054864, 0.0184697732, 0.8983123302, -0.2735522389, 0.0198625680, 0.0048169913, 0.0063736225, -0.0436291844, 0.0017527313, -0.0020891065, -0.0741934851, 10.8376998901, 0.9294568300, -0.0245931894, 5.3678073883, -0.1820224077, 0.0186468605, 0.8869744539, -0.2769004405, 0.0169417039, 0.0061308001, 0.0051326095, -0.1046696827, 0.0013565057, -0.0016542902, -0.0599494912, 13.1254348755, 0.7449229360, -0.0328789949, 6.0329661369, -0.1554213911, 0.0201931614, 0.8650057316, -0.2108421922, 0.0137868915, 0.0076632188, 0.0043527437, -0.1503998339, 0.0008775968, -0.0010392433, -0.0397608243, 15.1640777588, 0.5270256996, -0.0317670554, 6.7511920929, -0.1200928167, 0.0205541570, 0.8261252046, -0.1296072900, 0.0106071625, 0.0091231298, 0.0036226772, -0.1855581999, 0.0005234876, -0.0006949716, -0.0212099589, 0.6031303406, -0.5372571945, -5.5857901573, -0.8965449333, -2.8110265732, -2.5804800987, -1.3669095039, -2.8156251907, -2.7502050400, -6.9062500000, -6.9062500000, -2.5056722164, -5.9377503395, -5.8685836792, -4.5113325119, 0.5998771191, -0.5305532217, -5.4034900665, -0.9038143158, -2.8200101852, -2.5783247948, -1.3629288673, -2.8145039082, -2.7512984276, -6.9062495232, -6.9062495232, -2.5044553280, -5.9485292435, -5.8772983551, -4.5153522491, 0.5904710293, -0.5346295834, -4.9999585152, -0.8997977972, -2.8332068920, -2.5808563232, -1.3871815205, -2.8392422199, -2.7506220341, -6.9104094505, -6.9099307060, -2.5028140545, -5.9189891815, -5.8714599609, -4.5410609245, 0.5773304701, -0.5407657623, -4.5667905807, -0.9007066488, -2.8468425274, -2.5842781067, -1.3976426125, -2.8509068489, -2.7454934120, -6.8588881493, -6.8575105667, -2.5074367523, -5.9061331749, -5.8537712097, -4.5440773964, 0.5736197829, -0.5522801876, -4.1560473442, -0.9116510153, -2.8706526756, -2.5879228115, -1.4068591595, -2.8797378540, -2.7677533627, -6.6865072250, -6.6640448570, -2.4970521927, -5.9050121307, -5.8443865776, -4.5781807899, 0.5711638927, -0.5580114126, -3.7945392132, -0.9200813770, -2.8953692913, -2.5973043442, -1.4074099064, -2.8866109848, -2.7733435631, -6.5516653061, -6.5545163155, -2.4767024517, -5.9407577515, -5.8477673531, -4.5785651207, 0.5739260912, -0.5511945486, -3.5097734928, -0.9214478731, -2.9002292156, -2.6172537804, -1.4231437445, -2.8739135265, -2.7447261810, -6.5046043396, -6.4951958656, -2.4839122295, -5.9740934372, -5.8275337219, -4.6100416183, 0.5804507136, -0.5575226545, -3.2413945198, -0.9266450405, -2.9230244160, -2.6184477806, -1.4406759739, -2.9210374355, -2.7597925663, -6.4830093384, -6.4082822800, -2.4646644592, -6.0146675110, -5.8745117188, -4.6347498894, 0.5758368969, -0.5753260851, -3.0226893425, -0.9353455305, -2.9384720325, -2.6071338654, -1.4546266794, -2.9336247444, -2.7791597843, -6.4727578163, -6.2868704796, -2.4541912079, -6.0553798676, -5.9013609886, -4.6538867950, 0.5852775574, -0.5604020953, -2.8184165955, -0.9293215871, -2.9313814640, -2.6165087223, -1.4817104340, -2.9584798813, -2.7687325478, -6.4412493706, -6.1362581253, -2.4442248344, -6.0786633492, -5.8899812698, -4.6658864021, 0.5793739557, -0.5586300492, -2.6225042343, -0.9169694185, -2.9580035210, -2.6230833530, -1.4928466082, -2.9623339176, -2.7920286655, -6.4018325806, -6.0135512352, -2.4168140888, -6.1105055809, -5.8970222473, -4.6711778641, 0.5877459049, -0.5433277488, -2.4486436844, -0.8902211189, -2.9292149544, -2.6218116283, -1.4875974655, -2.9638609886, -2.7928924561, -6.3461632729, -5.8824586868, -2.4233117104, -6.1269531250, -5.8879561424, -4.6472301483, 0.6165555120, -0.5267498493, -2.2800240517, -0.8434946537, -2.9228394032, -2.6158106327, -1.4791308641, -2.9421224594, -2.8017868996, -6.2792358398, -5.7584309578, -2.4015488625, -6.1462097168, -5.8672728539, -4.5985975266, 0.6212996244, -0.5088018179, -2.1447784901, -0.7882515192, -2.8908855915, -2.6131021976, -1.4381370544, -2.8960034847, -2.7918782234, -6.1989512444, -5.6120390892, -2.3802313805, -6.1507153511, -5.8529963493, -4.5321588516, 0.6491242051, -0.4768840075, -2.0158619881, -0.7238204479, -2.8549571037, -2.6101040840, -1.3672302961, -2.8145575523, -2.7627944946, -6.1176156998, -5.4407706261, -2.3841636181, -6.1093335152, -5.7783207893, -4.4514441490, 0.6818878055, -0.4345123172, -1.8893152475, -0.6472602487, -2.7795901299, -2.5984642506, -1.2374693155, -2.7073123455, -2.6863570213, -6.0225930214, -5.2563080788, -2.3596868515, -6.0174131393, -5.6906609535, -4.3100142479, 0.7230042815, -0.4036316276, -1.7877944708, -0.5641269684, -2.6845011711, -2.5887403488, -1.0794155598, -2.5572817326, -2.6305654049, -5.9021773338, -5.0917840004, -2.3353469372, -5.9429283142, -5.5985307693, -4.1430716515, 0.7661570311, -0.3499242067, -1.6635777950, -0.4499340653, -2.5934765339, -2.5717239380, -0.9381453991, -2.4088954926, -2.5657043457, -5.7488989830, -4.9140653610, -2.3118832111, -5.7957038879, -5.4619922638, -3.9852266312, 0.8351538181, -0.3160147071, -1.5560088158, -0.3227483034, -2.4450032711, -2.5480890274, -0.8184899688, -2.2169010639, -2.5002965927, -5.5759687424, -4.7557630539, -2.2542095184, -5.6466512680, -5.3012876511, -3.8111629486, 0.9085252881, -0.2260122299, -1.4559940100, -0.1782942116, -2.3052258492, -2.5222449303, -0.7219318151, -2.0514018536, -2.4233548641, -5.4153981209, -4.5771903992, -2.1902751923, -5.5017485619, -5.1600246429, -3.6130099297, 0.9858909845, -0.1258360147, -1.3669396639, -0.0240099132, -2.1519875526, -2.4891524315, -0.6312087774, -1.8284138441, -2.3761672974, -5.2547063828, -4.4153099060, -2.1094584465, -5.3285489082, -4.9879822731, -3.4238064289, 1.0991904736, -0.0049605370, -1.2891969681, 0.1238748729, -1.9883587360, -2.4406855106, -0.5637221336, -1.5810493231, -2.2766506672, -5.0649743080, -4.2553782463, -2.0327672958, -5.1212849617, -4.8445420265, -3.1851806641, 1.1861916780, 0.1493574977, -1.1834793091, 0.2626062930, -1.7970973253, -2.3906242847, -0.5264706016, -1.2807112932, -2.2123823166, -4.8792576790, -4.1145439148, -1.9115428925, -4.9507217407, -4.7239751816, -2.9034457207, 1.2947194576, 0.3143202066, -1.0863986015, 0.3728095889, -1.5648000240, -2.3496675491, -0.5354782343, -1.0032274723, -2.1455500126, -4.6932187080, -4.0049867630, -1.7385971546, -4.7941389084, -4.6026549339, -2.6543893814, 1.4133480787, 0.5326241255, -0.9878123403, 0.4622476101, -1.3328492641, -2.3235943317, -0.5466824174, -0.8028020263, -2.0743997097, -4.5295057297, -3.9151287079, -1.5265369415, -4.6554141045, -4.4944581985, -2.4356999397, 1.5198161602, 0.8058823347, -0.8700630069, 0.5236044526, -1.1846492290, -2.2897078991, -0.5613500476, -0.6492928267, -2.0235350132, -4.3547496796, -3.8198957443, -1.2760257721, -4.5354199409, -4.4085898399, -2.2827630043, 1.6140869856, 1.0595301390, -0.7682074308, 0.5881240368, -1.0702744722, -2.2615752220, -0.5625019073, -0.5837324858, -1.9972016811, -4.1906595230, -3.7459712029, -1.0983450413, -4.4565920830, -4.3610162735, -2.2315008640, 1.7217743397, 1.3039419651, -0.6463991404, 0.6115790606, -1.0309325457, -2.2438652515, -0.6020388603, -0.5342514515, -1.9661116600, -4.0478649139, -3.6445455551, -0.9551031590, -4.4211630821, -4.3455538750, -2.2556293011, 1.7942794561, 1.5587598085, -0.5134651661, 0.6216995716, -1.0073661804, -2.2451047897, -0.6510437727, -0.5645346642, -1.9176748991, -3.9113125801, -3.5869798660, -0.8330588341, -4.3580846786, -4.3209800720, -2.3259627819, 1.8677375317, 1.7617244720, -0.4004968405, 0.6668716669, -0.9703812599, -2.2202031612, -0.6875780821, -0.6060823202, -1.8983026743, -3.7947616577, -3.5234701633, -0.7740106583, -4.3800630569, -4.3247947693, -2.3797197342, 1.9688360691, 1.9587824345, -0.2616862059, 0.7270054817, -0.8677362204, -2.2461016178, -0.7125653625, -0.6356588602, -1.8761273623, -3.6969804764, -3.5074410439, -0.7504589558, -4.4421920776, -4.3502469063, -2.4321258068, 2.0565929413, 2.1529774666, -0.1285124421, 0.7589756250, -0.8045961857, -2.2678759098, -0.7259448171, -0.7318037152, -1.8826504946, -3.6470680237, -3.5005617142, -0.7498800755, -4.4850792885, -4.4060864449, -2.5797433853, 2.1561298370, 2.3252487183, 0.0316394567, 0.8051829934, -0.7425311804, -2.2874979973, -0.7288597822, -0.8151277304, -1.9067963362, -3.6045622826, -3.4799845219, -0.7778179646, -4.5462536812, -4.4752783775, -2.7490820885, -15.7720756531, -0.8712290525, -0.1669511050, 0.0000516183, -0.2730379105, -0.0080998614, -0.0007910251, 0.0754788965, -0.0151181072, -0.0212875921, 0.0000492940, 0.0003367066, -0.0039356728, 0.0008115936, 0.0046383375, -0.0010997280, -0.8854498863, -0.1681820899, 0.0000329338, -0.2575283051, -0.0108742304, -0.0005422449, 0.0721801966, -0.0179612227, -0.0178367775, 0.0000617526, 0.0003869095, -0.0039264495, 0.0008104991, 0.0043745423, -0.0010398997, -0.9105464220, -0.1673723161, -0.0000430062, -0.2691574097, -0.0110023692, 0.0001566338, 0.0887122601, -0.0184433945, -0.0084507857, 0.0000926413, 0.0005400303, -0.0040430366, 0.0007480102, 0.0036785137, -0.0008968741, -0.9222247601, -0.1651731133, -0.0000045088, -0.2411537170, -0.0109238243, 0.0016300939, 0.0896726325, -0.0167001523, 0.0024922446, 0.0001291978, 0.0007076084, -0.0042338623, 0.0005835978, 0.0023500859, -0.0005321382, -0.9522330761, -0.1674269438, -0.0000445619, -0.2582216263, -0.0097667910, 0.0014159749, 0.0972671807, -0.0123001598, 0.0159865022, 0.0001649753, 0.0007713476, -0.0042323102, 0.0003220849, 0.0005657579, -0.0002537125, -0.9970006943, -0.1663401872, -0.0000044086, -0.2811431885, -0.0132765006, 0.0005608993, 0.0967902839, -0.0105969217, 0.0226335712, 0.0001574172, 0.0006758652, -0.0046055624, -0.0000291580, -0.0007417700, 0.0001723125, -1.0412650108, -0.1678341031, 0.0002765954, -0.2620410919, -0.0132951066, 0.0018973504, 0.1055460274, -0.0044726618, 0.0194410440, 0.0001492158, 0.0005154439, -0.0046349596, -0.0002842621, -0.0006875073, 0.0007801775, -1.0847778320, -0.1708790809, 0.0009773148, -0.1634120941, -0.0121687697, 0.0017377681, 0.1182145774, 0.0031424109, 0.0093229292, 0.0001239336, 0.0004936849, -0.0047828066, -0.0001625130, -0.0000892830, 0.0012966017, -1.1423377991, -0.1744516045, 0.0018219596, -0.1943655014, -0.0132084722, 0.0016226657, 0.1363812238, -0.0025610412, 0.0036798455, 0.0000676094, 0.0005591742, -0.0045677223, 0.0000298904, -0.0001173376, 0.0019804630, -1.1831550598, -0.1788347065, 0.0028479341, -0.1185846329, -0.0101548415, 0.0013207248, 0.1615974605, 0.0096578505, 0.0046132896, 0.0000921040, 0.0005146808, -0.0043211589, 0.0001450987, -0.0004170045, 0.0028638151, -1.2336597443, -0.1903054714, 0.0030200975, -0.1020088196, -0.0097770579, 0.0012420658, 0.1771132350, 0.0098006986, 0.0065152431, 0.0001927901, 0.0003896027, -0.0036500923, 0.0002054665, -0.0004544638, 0.0036507100, -1.2583751678, -0.1922641993, 0.0051896004, -0.0779085159, -0.0124815498, 0.0005310744, 0.2114773393, 0.0220001210, 0.0033983325, 0.0002234471, 0.0003255671, -0.0028665911, 0.0003396386, -0.0002873618, 0.0045666713, -1.2797603607, -0.1931598783, 0.0063287988, -0.0423288345, -0.0104200821, -0.0000328997, 0.2664591670, 0.0305477995, 0.0007096350, 0.0002880892, 0.0002548433, -0.0012317358, 0.0004526502, -0.0001316528, 0.0057355179, -1.2984199524, -0.2024210989, 0.0066664722, 0.0271663666, -0.0083047375, 0.0001823619, 0.3240397274, 0.0256734826, -0.0008861674, 0.0005651489, 0.0003010059, 0.0004673890, 0.0005108927, 0.0002274574, 0.0068862182, -1.2395801544, -0.2094673514, 0.0081634168, 0.0893402100, -0.0125560341, -0.0009688148, 0.4038239419, 0.0439986736, -0.0033949744, 0.0008077636, 0.0005081398, 0.0027164789, 0.0006419505, 0.0007386791, 0.0077472636, -1.2233295441, -0.2109905183, 0.0069417153, 0.2224941254, -0.0121149011, -0.0022782874, 0.4805579484, 0.0405384488, -0.0052230810, 0.0011056508, 0.0009624315, 0.0048312424, 0.0006715900, 0.0013187134, 0.0079018455, -1.1289825439, -0.2110992968, 0.0060278010, 0.4027118683, -0.0177147482, -0.0018596246, 0.5665094256, 0.0239039138, -0.0054305997, 0.0015075273, 0.0016418518, 0.0061184233, 0.0008156388, 0.0018480971, 0.0046239137, -1.0499534607, -0.2156256437, 0.0034687631, 0.6180763245, -0.0304864496, -0.0013234981, 0.6489440799, -0.0322717875, -0.0060219727, 0.0019604298, 0.0025962023, 0.0053557716, 0.0009888830, 0.0021575275, -0.0085061695, -0.8474082947, -0.2324554324, 0.0000870787, 0.8651752472, -0.0690926090, 0.0000024443, 0.7150048018, -0.1421115100, -0.0052532423, 0.0023565106, 0.0035689147, -0.0003044310, 0.0008820594, 0.0022281096, -0.0342857279, -0.5193519592, -0.2676124573, -0.0042655561, 1.1464891434, -0.1239956319, 0.0020395734, 0.7405831814, -0.2697045505, -0.0005995568, 0.0026400436, 0.0044604940, -0.0178285241, 0.0006861726, 0.0018644520, -0.0653255135, -0.0982742310, -0.3637424707, -0.0094874911, 1.4541559219, -0.1831470430, 0.0038764402, 0.7168091536, -0.3844248056, 0.0013321177, 0.0025203112, 0.0053686304, -0.0507663898, 0.0001946027, 0.0010938580, -0.0954079255, 0.5802497864, -0.5524717569, -0.0127265928, 1.7903690338, -0.2413132936, 0.0058255671, 0.6772662401, -0.4946811199, 0.0040041190, 0.0019192244, 0.0059148441, -0.0985724404, -0.0000369086, 0.0002036561, -0.1211472675, 1.3084449768, -0.8728629351, -0.0229594205, 2.0994434357, -0.2789647877, 0.0060451888, 0.6178426743, -0.5672478676, 0.0081972852, 0.0007679214, 0.0058221566, -0.1579641998, 0.0002498280, -0.0006604641, -0.1385802180, 2.3533706665, -1.3363046646, -0.0324588083, 2.4160366058, -0.2966139317, 0.0072213593, 0.5857231021, -0.6110146046, 0.0124607347, -0.0001766572, 0.0053847497, -0.2225791067, 0.0010470197, -0.0010883269, -0.1453436911, 3.4065017700, -1.9994001389, -0.0431961976, 2.7848854065, -0.2791729867, 0.0061766445, 0.6007498503, -0.5834974647, 0.0168752596, -0.0005232651, 0.0046075210, -0.2895481586, 0.0016444369, -0.0011657948, -0.1373186409, 4.7563323975, -2.8527562618, -0.0513686836, 3.1918992996, -0.2251515687, 0.0062165051, 0.6333975792, -0.4839854538, 0.0204584431, 0.0000696703, 0.0039834394, -0.3534365594, 0.0016821681, -0.0007237566, -0.1108013466, 6.0474853516, -3.8504118919, -0.0510942191, 3.5830411911, -0.1466991007, 0.0076697916, 0.6862339973, -0.2713772058, 0.0196774844, 0.0014322300, 0.0042286422, -0.4034142494, 0.0010215282, -0.0002725128, -0.0688805580, 7.4510345459, -5.0086097717, -0.0585790500, 4.0385408401, -0.0587114543, 0.0092953388, 0.7533562183, -0.0227043740, 0.0178342108, 0.0029275906, 0.0047727888, -0.4319105148, -0.0000312916, -0.0003220037, -0.0191604830, 8.6838607788, -6.3459315300, -0.0548461825, 4.5389490128, 0.0212593190, 0.0117559833, 0.7885663509, 0.2060831487, 0.0170537177, 0.0043732342, 0.0053206976, -0.4332718849, -0.0008063724, -0.0004263903, 0.0271582250, 10.4140701294, -7.6644124985, -0.0555476695, 5.0738067627, 0.0775867701, 0.0134549569, 0.7766780257, 0.3537742496, 0.0146836527, 0.0052827736, 0.0053938613, -0.4122793674, -0.0006474766, -0.0007480962, 0.0582527816, 11.8082504272, -8.9907979965, -0.0490981340, 5.5333366394, 0.1022725403, 0.0150432177, 0.7273942232, 0.3942302167, 0.0113843875, 0.0061575691, 0.0051175961, -0.3770926893, -0.0003476655, -0.0009511460, 0.0672562793, 13.5224990845, -10.2500543594, -0.0473378822, 5.9782629013, 0.0998832285, 0.0159309115, 0.6778737903, 0.3615520298, 0.0105361957, 0.0068121380, 0.0042947484, -0.3407532871, 0.0000489522, -0.0010689115, 0.0620895922, 15.1312408447, -11.5921106339, -0.0299314409, 6.4488868713, 0.0891876668, 0.0159213133, 0.6333137751, 0.2793364823, 0.0093284622, 0.0076149572, 0.0035739192, -0.3111143708, 0.0003715468, -0.0008036389, 0.0482363068, 0.6005553007, -0.5940812826, -5.6369757652, -0.7893590927, -2.6748538017, -2.6082596779, -1.4463723898, -2.7828617096, -2.6709532738, -6.9062500000, -6.9062500000, -2.5114166737, -6.0503072739, -5.9022789001, -4.3698468208, 0.5979242325, -0.5966616869, -5.4237689972, -0.7740852237, -2.6731700897, -2.6008906364, -1.4607176781, -2.7837128639, -2.6688568592, -6.9062500000, -6.9062500000, -2.5030550957, -6.0394124985, -5.8951663971, -4.3717513084, 0.6006034017, -0.6009682417, -4.9986233711, -0.7842100263, -2.6686179638, -2.6052055359, -1.4877002239, -2.7866272926, -2.6854188442, -6.9115619659, -6.9065237045, -2.5099220276, -5.9897074699, -5.8722372055, -4.3682904243, 0.6163814068, -0.5864676237, -4.5871739388, -0.7717229128, -2.6763384342, -2.6025409698, -1.5036188364, -2.7942225933, -2.6742696762, -6.9013342857, -6.8714613914, -2.4938552380, -5.9367527962, -5.8069705963, -4.3486104012, 0.6347873807, -0.5892113447, -4.1838746071, -0.7541753650, -2.6649560928, -2.5995025635, -1.4817911386, -2.8025498390, -2.6917490959, -6.7148370743, -6.6626372337, -2.4875562191, -5.9126887321, -5.7333426476, -4.3072671890, 0.6391712427, -0.5756171942, -3.8286147118, -0.7323865294, -2.6362142563, -2.6002771854, -1.4508426189, -2.7915806770, -2.6854743958, -6.5684680939, -6.5222005844, -2.4677352905, -5.9148526192, -5.6401062012, -4.2761483192, 0.6599438190, -0.5791758299, -3.5315828323, -0.7023681402, -2.6074676514, -2.6174468994, -1.4388110638, -2.7350394726, -2.6465032101, -6.4999647141, -6.4215393066, -2.4477498531, -5.9243559837, -5.5581059456, -4.2216811180, 0.6740121245, -0.5593730211, -3.2721009254, -0.6655355692, -2.5714893341, -2.6108450890, -1.4176791906, -2.6996378899, -2.6503863335, -6.4798235893, -6.3097953796, -2.4343669415, -5.9015636444, -5.5988979340, -4.1483249664, 0.6862339377, -0.5484998226, -3.0386135578, -0.6237251759, -2.5313439369, -2.6106622219, -1.3843166828, -2.6599059105, -2.6589021683, -6.4638848305, -6.1523261070, -2.4045519829, -5.8778591156, -5.5994167328, -4.0922751427, 0.7245219946, -0.5324175358, -2.8345079422, -0.5781433582, -2.4849655628, -2.6112766266, -1.3460229635, -2.6267411709, -2.6375818253, -6.4059629440, -5.9770760536, -2.3729681969, -5.8440666199, -5.5536813736, -3.9951162338, 0.7383230925, -0.5187578797, -2.6435389519, -0.5241207480, -2.4233882427, -2.6070947647, -1.2972903252, -2.5387568474, -2.6430261135, -6.3233704567, -5.7820930481, -2.3406622410, -5.7981972694, -5.4995179176, -3.8958013058, 0.7606922984, -0.4901129603, -2.4749107361, -0.4476329088, -2.3627629280, -2.5893807411, -1.2323193550, -2.4431836605, -2.6307210922, -6.2224965096, -5.5902266502, -2.2996184826, -5.7286920547, -5.4377503395, -3.7581915855, 0.8026971817, -0.4497594833, -2.3061237335, -0.3949539959, -2.3034672737, -2.5918269157, -1.1313705444, -2.3320314884, -2.6105713844, -6.0969018936, -5.3820953369, -2.2391805649, -5.6244001389, -5.3302106857, -3.5896751881, 0.8497940302, -0.4037554264, -2.1523392200, -0.3031080961, -2.2174258232, -2.5797548294, -1.0388026237, -2.1615722179, -2.5721702576, -5.9304151535, -5.1978764534, -2.1974556446, -5.5156512260, -5.2140359879, -3.3803219795, 0.8896974921, -0.3420684338, -2.0117928982, -0.2238227576, -2.1022996902, -2.5560681820, -0.9405190945, -1.9506516457, -2.5097465515, -5.7640724182, -4.9965124130, -2.1161892414, -5.4122996330, -5.1011638641, -3.1366119385, 0.9476752281, -0.2814202309, -1.8753561974, -0.1153686047, -1.9765152931, -2.5284936428, -0.8498904705, -1.6927145720, -2.4642186165, -5.5848703384, -4.8126511574, -2.0310151577, -5.2420749664, -4.9803023338, -2.8633568287, 1.0167870522, -0.1901880503, -1.7545940876, -0.0189817846, -1.7905375957, -2.4976537228, -0.7620817423, -1.3876794577, -2.3895361423, -5.3919510841, -4.6192092896, -1.9341650009, -5.0996623039, -4.8807177544, -2.5576050282, 1.0906511545, -0.0755019188, -1.6426146030, 0.0714819133, -1.5889346600, -2.4368081093, -0.6914750934, -1.0767412186, -2.3531336784, -5.2210764885, -4.4596452713, -1.7919135094, -4.9558267593, -4.7769699097, -2.2491352558, 1.1811416149, 0.0553742647, -1.5340439081, 0.1597722173, -1.3777644634, -2.4008226395, -0.6451447606, -0.8261585236, -2.2863693237, -5.0193667412, -4.3029613495, -1.6144804955, -4.7773509026, -4.6435480118, -1.9907000065, 1.2828088999, 0.2241292000, -1.4151123762, 0.2299863100, -1.2260870934, -2.3562235832, -0.6396774650, -0.6718767881, -2.1993517876, -4.8133621216, -4.1539082527, -1.4163622856, -4.6088576317, -4.5151081085, -1.8363945484, 1.3666030169, 0.4423956871, -1.3037776947, 0.2681757212, -1.1435334682, -2.3131146431, -0.6113398075, -0.5750358105, -2.1074607372, -4.6232428551, -3.9962701797, -1.2244317532, -4.4377498627, -4.3820972443, -1.7646765709, 1.4615591764, 0.6832343340, -1.1960432529, 0.3198612928, -1.1250317097, -2.2776019573, -0.5904153585, -0.5604701042, -2.0210955143, -4.4258322716, -3.8356385231, -1.0439579487, -4.3094363213, -4.2676229477, -1.7995171547, 1.5596895218, 0.9171035290, -1.0706846714, 0.3641597033, -1.1256451607, -2.2520937920, -0.5594606400, -0.5158802271, -1.9508979321, -4.2328877449, -3.6819841862, -0.9181363583, -4.2155709267, -4.1734275818, -1.8711247444, 1.6405692101, 1.1365606785, -0.9596373439, 0.4375356436, -1.0974373817, -2.2070448399, -0.5031561255, -0.4956446886, -1.9169826508, -4.0732297897, -3.5407619476, -0.8307461739, -4.1619505882, -4.1454529762, -1.9184856415, 1.7177228928, 1.3449796438, -0.8446478248, 0.5276324153, -1.0591233969, -2.1711606979, -0.4555291533, -0.4397404194, -1.8195052147, -3.9261667728, -3.4723715782, -0.7593383789, -4.1056318283, -4.0656771660, -1.9652369022, 1.7813019753, 1.5340373516, -0.7177729011, 0.6600973010, -1.0090471506, -2.1557378769, -0.4193481803, -0.3474802971, -1.7992926836, -3.8402690887, -3.3894038200, -0.7047679424, -4.1212186813, -4.0088920593, -1.9651656151, 1.8526307344, 1.7007281780, -0.6046216488, 0.7526525259, -0.9446974993, -2.1433598995, -0.4150798321, -0.2997372150, -1.7444374561, -3.7851548195, -3.3294365406, -0.6863148212, -4.1464548111, -3.9814379215, -1.9553103447, 1.9311887026, 1.8653805256, -0.4668562412, 0.8324231505, -0.9118283987, -2.1271162033, -0.4234728813, -0.2810773849, -1.7680326700, -3.7443265915, -3.3096742630, -0.6792161465, -4.2019605637, -4.0000376701, -1.9465260506, 2.0010602474, 2.0357766151, -0.3617584705, 0.9018946290, -0.8968369961, -2.1350927353, -0.4696878791, -0.3540768623, -1.7772755623, -3.6765017509, -3.3038837910, -0.6629173756, -4.2621283531, -4.0610570908, -2.0156064034, 2.0858235359, 2.1990861893, -0.2341583371, 0.9350270033, -0.9114624262, -2.1759934425, -0.4979435802, -0.4750006199, -1.8414580822, -3.5983662605, -3.3195424080, -0.6414980888, -4.3612003326, -4.1583409309, -2.1563158035, 2.1589543819, 2.3553035259, -0.0960173607, 0.9660774469, -0.8875650167, -2.1997237206, -0.5211155415, -0.6296023726, -1.8813668489, -3.5384356976, -3.3117175102, -0.6100690365, -4.4517664909, -4.2642364502, -2.3294942379, 2.2329230309, 2.5175871849, 0.0172773600, 0.9774711132, -0.8568613529, -2.2222108841, -0.5591777563, -0.7440729141, -1.9211559296, -3.5071005821, -3.2840332985, -0.5511507988, -4.5223731995, -4.3436779976, -2.5063571930, 2.3196201324, 2.6712112427, 0.1504655480, 1.0088610649, -0.8050366640, -2.2640442848, -0.5852780342, -0.7938364148, -1.9425649643, -3.4887850285, -3.2833657265, -0.5385830402, -4.5723748207, -4.4109268188, -2.6321842670, -16.1381626129, -0.8397234678, -0.1735909581, -0.0000666266, 0.0662746429, -0.0428304225, -0.0034279590, 0.6422700286, -0.1003720313, -0.0211206023, 0.0000977472, 0.0003760287, -0.0084245969, 0.0015898240, 0.0057927947, -0.0199869834, -0.8511806726, -0.1734850705, -0.0001009831, 0.0537014008, -0.0442098081, -0.0029174900, 0.6530832648, -0.1062603444, -0.0198831521, 0.0001222808, 0.0004377715, -0.0087611591, 0.0016212567, 0.0055801682, -0.0205264315, -0.8691308498, -0.1770244837, -0.0002325880, 0.0883064270, -0.0454724953, -0.0024873484, 0.6861693263, -0.1130906194, -0.0116151990, 0.0001788564, 0.0006282119, -0.0091117676, 0.0014972589, 0.0048835506, -0.0219666716, -0.8594462872, -0.1793617010, -0.0004159065, 0.1156034470, -0.0502690859, -0.0019210577, 0.7296526432, -0.1250580847, -0.0031027012, 0.0002639318, 0.0008757267, -0.0100421151, 0.0011423380, 0.0037291571, -0.0240657851, -0.8498423100, -0.1803146005, -0.0007580824, 0.1815738678, -0.0584708229, -0.0028814897, 0.7931434512, -0.1340679377, 0.0070392396, 0.0003431701, 0.0010444040, -0.0118381782, 0.0007996247, 0.0022431391, -0.0274528693, -0.8468608856, -0.1886191517, -0.0010954859, 0.2526826859, -0.0642152429, -0.0041545918, 0.8715334535, -0.1533855498, 0.0141144190, 0.0004112762, 0.0011226832, -0.0144179501, 0.0005435919, 0.0016373632, -0.0326269455, -0.8068876266, -0.1968971640, -0.0016249223, 0.3750944138, -0.0789209232, -0.0050442573, 0.9493745565, -0.1783989519, 0.0133262873, 0.0004524854, 0.0012277288, -0.0181677975, 0.0004410110, 0.0021481011, -0.0394462384, -0.7821383476, -0.2105256021, -0.0022428923, 0.5264072418, -0.0951290429, -0.0038848799, 1.0560929775, -0.2203049213, 0.0063196504, 0.0005207726, 0.0016238784, -0.0235843901, 0.0007479461, 0.0026840661, -0.0487335846, -0.7087898254, -0.2335432768, -0.0025309757, 0.7122049332, -0.1165380031, -0.0023392821, 1.1488356590, -0.2696315646, 0.0062458161, 0.0007145777, 0.0021983082, -0.0319227800, 0.0009987494, 0.0023369798, -0.0605352633, -0.6140575409, -0.2636965811, -0.0032360139, 1.0001688004, -0.1447826177, -0.0010076324, 1.2091524601, -0.3315420449, 0.0124507472, 0.0008171856, 0.0025665867, -0.0430649929, 0.0008228868, 0.0016291920, -0.0748889372, -0.4073247910, -0.3137666881, -0.0038195176, 1.2900953293, -0.1783401519, -0.0003142045, 1.2330539227, -0.3956319988, 0.0158201866, 0.0010879238, 0.0027953421, -0.0591641292, 0.0007062239, 0.0010785919, -0.0917896330, -0.1598749161, -0.3842948675, -0.0028900271, 1.6261558533, -0.2197511196, 0.0027437147, 1.2176456451, -0.4691663384, 0.0161637962, 0.0013880925, 0.0028519568, -0.0809112564, 0.0004721026, 0.0002433618, -0.1101897508, 0.1616783142, -0.4858596921, -0.0021727274, 1.9664993286, -0.2619256079, 0.0047567911, 1.1751297712, -0.5425533056, 0.0097814910, 0.0015265882, 0.0027403866, -0.1086055487, -0.0003068991, -0.0003314478, -0.1289717555, 0.6025505066, -0.6414657831, -0.0043031960, 2.2849302292, -0.2965071201, 0.0069139483, 1.1100498438, -0.6057927608, 0.0104599334, 0.0016882458, 0.0025652819, -0.1435588747, -0.0009529112, -0.0011083621, -0.1473649889, 1.1156044006, -0.8456279039, -0.0036315760, 2.6208066940, -0.3256083727, 0.0079654176, 1.0438309908, -0.6451689005, 0.0098670162, 0.0013460601, 0.0023203271, -0.1851063073, -0.0012099382, -0.0014806039, -0.1606637090, 1.7760448456, -1.1380245686, -0.0060937582, 2.9534940720, -0.3447786570, 0.0095148105, 0.9777711034, -0.6701621413, 0.0062041925, 0.0009638619, 0.0019770344, -0.2344535440, -0.0012848569, -0.0018033001, -0.1677632928, 2.4643688202, -1.5163282156, -0.0119488826, 3.3113832474, -0.3436095715, 0.0100079086, 0.9268333912, -0.6760493517, 0.0088852290, 0.0007098147, 0.0018778685, -0.2900279164, -0.0006701192, -0.0019477154, -0.1693762243, 3.2886085510, -2.0024743080, -0.0153172165, 3.6652173996, -0.3293984830, 0.0113014970, 0.8819763064, -0.6589868069, 0.0142467273, 0.0010296572, 0.0019219562, -0.3495438397, 0.0001442899, -0.0023649577, -0.1632654965, 4.2009544373, -2.6147477627, -0.0225742348, 4.0309457779, -0.3046873808, 0.0102605680, 0.8521239758, -0.6158993840, 0.0125715304, 0.0012886645, 0.0020021189, -0.4105588794, 0.0010146057, -0.0020383177, -0.1506769359, 5.1299591064, -3.3415138721, -0.0317798406, 4.4550476074, -0.2642821670, 0.0111501412, 0.8540866971, -0.5548675060, 0.0177458785, 0.0025735896, 0.0020519057, -0.4679658413, 0.0017235400, -0.0016201093, -0.1304942220, 6.2143325806, -4.1930122375, -0.0453531146, 4.8313426971, -0.2157468945, 0.0112467129, 0.8645417690, -0.4381200075, 0.0115326773, 0.0039854930, 0.0021624002, -0.5161566138, 0.0018561801, -0.0008715480, -0.1046467870, 7.3299217224, -5.1747107506, -0.0519666225, 5.1986637115, -0.1661562622, 0.0143460045, 0.8757280707, -0.2991250455, 0.0125130136, 0.0052887420, 0.0025875422, -0.5551793575, 0.0011226446, -0.0004611806, -0.0721129403, 8.5075492859, -6.2337183952, -0.0652396381, 5.5962657928, -0.1156491041, 0.0141661447, 0.8805187941, -0.1595472395, 0.0096496195, 0.0065953648, 0.0031231218, -0.5815792084, 0.0005322527, -0.0001365869, -0.0413452722, 9.7456626892, -7.4379329681, -0.0813413709, 5.9659452438, -0.0739623010, 0.0159171969, 0.8702517748, -0.0535392463, 0.0076262797, 0.0075944201, 0.0035583470, -0.5975923538, -0.0001808476, 0.0002814151, -0.0171252601, 10.8939437866, -8.7262811661, -0.0921076536, 6.3519515991, -0.0411724709, 0.0171423722, 0.8352737427, 0.0136737768, 0.0070646564, 0.0078528095, 0.0042951135, -0.6045275331, -0.0005957909, 0.0003282598, -0.0017840939, 12.3282470703, -10.1271505356, -0.1109456345, 6.6774458885, -0.0272827335, 0.0195370987, 0.7755869627, 0.0465720110, 0.0082089305, 0.0076473779, 0.0048272619, -0.6035295725, -0.0005773737, 0.0000613164, 0.0063495054, 13.6278381348, -11.6258850098, -0.1236564815, 7.0218114853, -0.0063749813, 0.0181997940, 0.7141370773, 0.0573468246, 0.0059001422, 0.0073209116, 0.0047591911, -0.5955991745, -0.0005622096, -0.0000117601, 0.0098937638, 15.1430816650, -13.2054805756, -0.1327949762, 7.3503041267, -0.0062282216, 0.0191860460, 0.6410401464, 0.0530951284, 0.0079827765, 0.0074445452, 0.0050083301, -0.5830495954, -0.0002900297, -0.0001862147, 0.0113324085, 16.8364028931, -14.8939123154, -0.1397719830, 7.7003479004, -0.0003857538, 0.0167240370, 0.5632885695, 0.0507920645, 0.0062606130, 0.0071846088, 0.0047072116, -0.5698649883, -0.0002158485, -0.0002065205, 0.0108011672, 18.3550720215, -16.6701354980, -0.1503064334, 7.9614720345, 0.0048053274, 0.0159543678, 0.4840753078, 0.0430551693, 0.0064645028, 0.0070666741, 0.0047130622, -0.5565985441, 0.0000018984, -0.0003553363, 0.0099561000, 20.0332565308, -18.5304260254, -0.1592886746, 8.1716022491, 0.0011685593, 0.0130681926, 0.3917582035, 0.0241467953, 0.0032860101, 0.0068631102, 0.0043157539, -0.5405481458, -0.0000222655, -0.0001550863, 0.0089699337, 21.8615341187, -20.3846950531, -0.1325635910, 8.3785629272, 0.0045432807, 0.0109330267, 0.3074857295, 0.0129718538, 0.0049510091, 0.0062264102, 0.0043141460, -0.5239577889, 0.0000168402, -0.0006260442, 0.0070432937, 23.7349014282, -22.2591342926, -0.1457871795, 8.5920839310, 0.0004147576, 0.0082399137, 0.2379673719, -0.0013800361, 0.0053276126, 0.0064245001, 0.0038683796, -0.5076529980, -0.0001474137, -0.0004002301, 0.0059532328, 0.6656712294, -0.5746699572, -5.5096487999, -0.3268334866, -1.7806559801, -2.4813585281, -0.9444276690, -1.5131449699, -2.5004820824, -6.9062495232, -6.9062495232, -2.3759925365, -5.4554982185, -5.3478279114, -3.0309278965, 0.6653420329, -0.5626199245, -5.2977561951, -0.3288486004, -1.7668770552, -2.4737176895, -0.9523570538, -1.5056178570, -2.4930953979, -6.9062495232, -6.9062495232, -2.3637845516, -5.4439086914, -5.3083658218, -3.0272614956, 0.6821708679, -0.5652115345, -4.8909468651, -0.3109810948, -1.7482632399, -2.4684205055, -0.9704904556, -1.4634058475, -2.4974722862, -6.9062676430, -6.9091310501, -2.3513975143, -5.3766107559, -5.2169241905, -2.9825177193, 0.7116826177, -0.5440986156, -4.4516324997, -0.2897558510, -1.7088379860, -2.4612181187, -0.9805006981, -1.4139428139, -2.4813830853, -6.8548965454, -6.8236188889, -2.3207297325, -5.2951545715, -5.0803537369, -2.9009521008, 0.7329712510, -0.5213584900, -4.0438051224, -0.2672377527, -1.6452926397, -2.4438202381, -0.9482175112, -1.3340477943, -2.4839196205, -6.6148452759, -6.5722417831, -2.2826523781, -5.2088994980, -4.9425530434, -2.8213193417, 0.7593109012, -0.4890168905, -3.6846961975, -0.2357007712, -1.5731351376, -2.4546732903, -0.9030162096, -1.2420623302, -2.4396750927, -6.3730254173, -6.2959990501, -2.2220840454, -5.1266946793, -4.8231544495, -2.7056882381, 0.7876214981, -0.4552334547, -3.3712029457, -0.1952591240, -1.4613600969, -2.4478764534, -0.8559556007, -1.1106306314, -2.3704867363, -6.1762523651, -5.9571542740, -2.1668970585, -5.0332169533, -4.7212262154, -2.5824952126, 0.8004986644, -0.3875161409, -3.1050009727, -0.1806978285, -1.3534115553, -2.4218719006, -0.8065717220, -0.9796406031, -2.3125774860, -5.9759855270, -5.6600704193, -2.0525527000, -4.9208812714, -4.6909608841, -2.4411733150, 0.8498231769, -0.3287036419, -2.8511760235, -0.1197116375, -1.2266398668, -2.3973197937, -0.7480068207, -0.8490550518, -2.2555143833, -5.7905607224, -5.3886270523, -1.9577004910, -4.8075175285, -4.6336698532, -2.3008205891, 0.8710952401, -0.2290375233, -2.6396944523, -0.0816056728, -1.1239068508, -2.3750705719, -0.7140636444, -0.7456262112, -2.2010226250, -5.5799689293, -5.1428251266, -1.8294517994, -4.6383647919, -4.5416669846, -2.1669073105, 0.9051587582, -0.1170188189, -2.4271659851, -0.0297439396, -1.0101188421, -2.3370590210, -0.6854567528, -0.6091728210, -2.1337637901, -5.3696708679, -4.9034886360, -1.6779620647, -4.5327849388, -4.4422445297, -2.0430815220, 0.9378940463, 0.0425328016, -2.2456476688, 0.0115647018, -0.9133206606, -2.3359296322, -0.6552662849, -0.4787878990, -2.0541622639, -5.1559123993, -4.6587247849, -1.5222539902, -4.4070611000, -4.3351192474, -1.9141237736, 0.9850500822, 0.2115888596, -2.0552082062, 0.0793885589, -0.8327689171, -2.2988371849, -0.6447067261, -0.3516204357, -1.9745497704, -4.9423341751, -4.4303407669, -1.3548967838, -4.2544760704, -4.2251691818, -1.8146073818, 1.0266213417, 0.4072366953, -1.8887705803, 0.1342371702, -0.7530275583, -2.2810521126, -0.6547397375, -0.2078495026, -1.8892548084, -4.7423863411, -4.2239723206, -1.2125110626, -4.1281657219, -4.1235303879, -1.7201743126, 1.0856002569, 0.6155968904, -1.7295880318, 0.1981676221, -0.6927821636, -2.2675168514, -0.6630265713, -0.0909037590, -1.8269014359, -4.5433917046, -4.0224962234, -1.0667016506, -4.0341706276, -4.0215640068, -1.5895822048, 1.1363761425, 0.8249385357, -1.5679652691, 0.2722414732, -0.6293085814, -2.2658181190, -0.6785701513, -0.0015695095, -1.7877076864, -4.3587970734, -3.8690752983, -0.9256622791, -3.9357388020, -3.9563579559, -1.5173802376, 1.2063299417, 1.0315783024, -1.4094810486, 0.3321238756, -0.5807471275, -2.2444939613, -0.7023128271, 0.0406010151, -1.7394971848, -4.1902027130, -3.7541947365, -0.7972819805, -3.9117522240, -3.9202651978, -1.4697604179, 1.2743554115, 1.2376940250, -1.2475583553, 0.3905830383, -0.5463316441, -2.2245061398, -0.7137642503, 0.0535819530, -1.7270593643, -4.0515851974, -3.6643331051, -0.6829016209, -3.9293289185, -3.8899335861, -1.4615836143, 1.3611469269, 1.4431896210, -1.1024882793, 0.4633552432, -0.5035657883, -2.2244038582, -0.7006895542, 0.0451915264, -1.7191816568, -3.9398810863, -3.5954740047, -0.6018171310, -3.9176921844, -3.8642878532, -1.5004346371, 1.4549303055, 1.6324951649, -0.9647514820, 0.5163282752, -0.4961724281, -2.2048051357, -0.6716566682, 0.0210351944, -1.7092759609, -3.8506093025, -3.5306191444, -0.5412843227, -3.9605031013, -3.8806853294, -1.5604915619, 1.5407100916, 1.8175170422, -0.8172644377, 0.5814472437, -0.4943002462, -2.1949832439, -0.6439061165, 0.0040770769, -1.7103087902, -3.7488989830, -3.4829442501, -0.4874989986, -3.9925491810, -3.8660316467, -1.6091647148, 1.6608301401, 2.0233671665, -0.6646399498, 0.6495342255, -0.4829334021, -2.2048211098, -0.6487714052, -0.0598739386, -1.7181918621, -3.7017765045, -3.4208393097, -0.4248068333, -4.0719046593, -3.9249770641, -1.6635880470, 1.7571657896, 2.1921401024, -0.5234251022, 0.6963846684, -0.4805669785, -2.2300989628, -0.6721278429, -0.1354691982, -1.7819347382, -3.6363337040, -3.3735039234, -0.3912503719, -4.1693439484, -3.9791183472, -1.7729754448, 1.8692162037, 2.3739748001, -0.3908829689, 0.7518083453, -0.4541776180, -2.2464740276, -0.7047370672, -0.2753115892, -1.8320609331, -3.5955977440, -3.3199138641, -0.3392944336, -4.2395825386, -4.0760993958, -1.9132685661, 1.9795069695, 2.5359733105, -0.2462549210, 0.7837955952, -0.4414095879, -2.2706847191, -0.6982480288, -0.4521298409, -1.8841862679, -3.5682406425, -3.2960064411, -0.2811379433, -4.3436393738, -4.1464567184, -2.1445477009, 2.0872659683, 2.7038285732, -0.1254863739, 0.8237404823, -0.4330542088, -2.2986903191, -0.7187151313, -0.6163543463, -1.8987168074, -3.5385065079, -3.2691853046, -0.2216012478, -4.4035701752, -4.2451105118, -2.3909676075, 2.2063243389, 2.8740415573, 0.0207754374, 0.8601200581, -0.3989268541, -2.3460941315, -0.7580927014, -0.7715440989, -1.9390017986, -3.5218830109, -3.2335186005, -0.2221171856, -4.4644112587, -4.2634429932, -2.6033208370, 2.3083646297, 3.0052268505, 0.1498959064, 0.8868039846, -0.3830972910, -2.3646051884, -0.7580697536, -0.8594346046, -1.9355098009, -3.5022947788, -3.1949870586, -0.1823587418, -4.4834427834, -4.3339171410, -2.7795658112, 2.4124412537, 3.1564397812, 0.2764173746, 0.9182792306, -0.3481491804, -2.3982355595, -0.7925620079, -0.9366729856, -1.9465967417, -3.4930422306, -3.1642184258, -0.1761841774, -4.5300984383, -4.3382225037, -2.9083316326, 2.5033330917, 3.2707540989, 0.4083464146, 0.9161332250, -0.3349491358, -2.4179801941, -0.8063015938, -0.9948927164, -1.9294482470, -3.4592666626, -3.1298019886, -0.1797850132, -4.5518832207, -4.3738636971, -2.9912879467, 2.6024372578, 3.3961193562, 0.5389289856, 0.9299134016, -0.3269613981, -2.4196345806, -0.8086945415, -1.0051964521, -1.9484823942, -3.4453747272, -3.1298177242, -0.1899013519, -4.5644226074, -4.3931875229, -3.0520360470, 2.6837472916, 3.5090243816, 0.6490012407, 0.9309875369, -0.3095061779, -2.4556770325, -0.7838326693, -0.9861089587, -1.9142237902, -3.4438090324, -3.1233010292, -0.2400543690, -4.5813937187, -4.3905830383, -3.0759391785, 2.7790641785, 3.6149787903, 0.7711520195, 0.9522712231, -0.2942028046, -2.4818992615, -0.7589547634, -0.9715220332, -1.8890123367, -3.4265625477, -3.1197319031, -0.2536659241, -4.5441865921, -4.3997898102, -3.0687968731, -16.3858642578, -0.7960101962, -0.1675691456, -0.0000223085, -0.3716650009, -0.0084207505, -0.0017275861, -0.0455610715, -0.0093217082, -0.0118046664, 0.0000491087, 0.0003159462, -0.0034500896, 0.0008694205, 0.0042258636, -0.0005293700, -0.8068246841, -0.1676158607, -0.0000371264, -0.3652830124, -0.0087207854, -0.0017941704, -0.0485307574, -0.0104918461, -0.0083330665, 0.0000619884, 0.0003666569, -0.0035617435, 0.0008825331, 0.0040032682, -0.0005321028, -0.8283030987, -0.1654621363, -0.0001237815, -0.3585243225, -0.0078991055, -0.0005082577, -0.0427460149, -0.0059339022, -0.0016834801, 0.0001012258, 0.0005132129, -0.0034770472, 0.0008131249, 0.0032759134, -0.0004219463, -0.8352439404, -0.1680109501, -0.0002198018, -0.3798618317, -0.0073475121, 0.0009656254, -0.0443360806, -0.0061531141, 0.0114371367, 0.0001351302, 0.0006784204, -0.0035240147, 0.0006590206, 0.0020029843, -0.0001139916, -0.8588075638, -0.1678876877, -0.0002702336, -0.3766317368, -0.0080586597, 0.0014185132, -0.0400466993, -0.0045992155, 0.0205757152, 0.0001751953, 0.0007215865, -0.0035833768, 0.0004304473, 0.0004269643, 0.0000433843, -0.9013118744, -0.1711971760, -0.0001308550, -0.3887224197, -0.0078450926, 0.0014480988, -0.0435044244, -0.0019475990, 0.0241778735, 0.0002008596, 0.0005949873, -0.0037425435, 0.0001528746, -0.0006097920, 0.0005364245, -0.9478254318, -0.1701239496, 0.0000068815, -0.3968887329, -0.0067552291, 0.0015746367, -0.0524200574, 0.0001700744, 0.0185421500, 0.0002068634, 0.0004497154, -0.0036685751, 0.0000261961, -0.0005711604, 0.0009817091, -1.0120277405, -0.1699153036, 0.0009602874, -0.3856430054, -0.0055623697, 0.0017940667, -0.0564941540, 0.0052871136, 0.0097050071, 0.0001872434, 0.0004326302, -0.0034939488, 0.0001374223, -0.0000134859, 0.0013343910, -1.0799980164, -0.1735107452, 0.0019549429, -0.4062385559, -0.0058359914, 0.0020222617, -0.0471655615, 0.0109018032, 0.0035396433, 0.0002029677, 0.0004879035, -0.0037255059, 0.0002421669, 0.0001063335, 0.0018703885, -1.1654605865, -0.1767468452, 0.0026437901, -0.4558219910, -0.0062979236, 0.0012624448, -0.0360553637, 0.0117447358, 0.0053554177, 0.0002397927, 0.0005298450, -0.0034545069, 0.0002183899, 0.0000254118, 0.0024027922, -1.2783985138, -0.1802948713, 0.0041604694, -0.4816522598, -0.0042761001, 0.0007499645, -0.0280169677, 0.0138479900, 0.0063060764, 0.0002634099, 0.0005201375, -0.0031156193, 0.0002152974, 0.0000531532, 0.0030583402, -1.3868713379, -0.1811879575, 0.0059476793, -0.4737253189, -0.0044669430, 0.0011524726, -0.0221739970, 0.0197334662, 0.0026575099, 0.0003103424, 0.0005030838, -0.0022594873, 0.0002201000, 0.0000961899, 0.0033840307, -1.4901542664, -0.1782621443, 0.0076636439, -0.4593296051, -0.0037159026, 0.0012814787, -0.0043193102, 0.0191348176, 0.0003369057, 0.0004034030, 0.0004742688, -0.0010507435, 0.0002746808, 0.0000665749, 0.0037355027, -1.6119632721, -0.1780516356, 0.0107436767, -0.4711170197, -0.0043401853, 0.0011242512, 0.0108286217, 0.0157261342, 0.0011237345, 0.0003999688, 0.0005434284, 0.0000585829, 0.0002553199, 0.0001048354, 0.0037409966, -1.7394981384, -0.1774877012, 0.0120298322, -0.4824571609, -0.0032532979, 0.0007781459, 0.0114605278, 0.0167344585, -0.0005522245, 0.0004638286, 0.0005664648, 0.0017358921, 0.0001864888, 0.0001051559, 0.0037152271, -1.8817272186, -0.1728899777, 0.0153629873, -0.4913978577, -0.0020630192, 0.0003582325, 0.0177655891, 0.0120180352, -0.0012677544, 0.0005247853, 0.0006809254, 0.0033140406, 0.0001971044, 0.0001416487, 0.0033591730, -2.0016098022, -0.1654484570, 0.0177684724, -0.4770746231, -0.0036749877, 0.0007189356, 0.0094167814, 0.0104107624, -0.0016356523, 0.0005389759, 0.0008133895, 0.0049842596, 0.0001548130, 0.0001494026, 0.0030465159, -2.1197624207, -0.1610096097, 0.0200309176, -0.4654245377, -0.0028719772, 0.0007871001, -0.0043295622, 0.0069569871, -0.0021029196, 0.0006770968, 0.0008426871, 0.0069921836, 0.0001214537, 0.0001279598, 0.0026764814, -2.2836875916, -0.1592709422, 0.0196048468, -0.4875307083, -0.0024766503, 0.0012189215, -0.0134110302, 0.0058473814, -0.0013114878, 0.0005941545, 0.0009259621, 0.0093806842, 0.0000854810, 0.0000644838, 0.0023403666, -2.4376411438, -0.1654946506, 0.0214534588, -0.5160055161, -0.0023443447, 0.0012299730, -0.0075379983, 0.0031579509, -0.0016639000, 0.0006134186, 0.0009389449, 0.0119415652, 0.0000241817, 0.0001235198, 0.0020060386, -2.5528907776, -0.1821739674, 0.0210555270, -0.5426187515, -0.0011044480, 0.0011125375, 0.0105880275, 0.0032345648, -0.0012190199, 0.0006175132, 0.0010377968, 0.0142241465, -0.0000186468, 0.0002041859, 0.0021417669, -2.8088493347, -0.1864134967, 0.0196865872, -0.5450105667, -0.0022472842, 0.0014884630, 0.0439582467, 0.0024227425, -0.0007883799, 0.0006211699, 0.0012498026, 0.0166206658, -0.0000079321, 0.0002743954, 0.0023565376, -2.9343643188, -0.1973351836, 0.0188050941, -0.5332908630, -0.0025655469, 0.0016313666, 0.0734919608, -0.0006011613, -0.0018622574, 0.0006740313, 0.0015557217, 0.0188534260, 0.0000196821, 0.0002531004, 0.0022815885, -3.0393295288, -0.2120441794, 0.0177504197, -0.4710197449, -0.0038036807, 0.0016670031, 0.0842305124, -0.0032999441, -0.0015003697, 0.0006270958, 0.0017661453, 0.0211944487, -0.0000010458, 0.0002935822, 0.0022293488, -3.2935714722, -0.2332189679, 0.0194917023, -0.4483366013, -0.0052063195, 0.0008294489, 0.1010637283, -0.0032167658, -0.0017291778, 0.0005172819, 0.0017448887, 0.0240284372, -0.0000222301, 0.0002407975, 0.0024751017, -3.4165267944, -0.2491450310, 0.0218490511, -0.3645191193, -0.0049766935, -0.0001807776, 0.1071538627, -0.0041939616, -0.0022568577, 0.0003737967, 0.0019687347, 0.0273308828, 0.0000093324, 0.0001807052, 0.0025963937, -3.4078750610, -0.2708975077, 0.0282736570, -0.3011798859, -0.0054350495, -0.0000829846, 0.1001108140, -0.0034665447, -0.0004182856, 0.0003475213, 0.0018572948, 0.0307642501, 0.0000050145, 0.0001308546, 0.0026692087, -3.5052413940, -0.2742791176, 0.0283154845, -0.2465648651, -0.0076353732, -0.0003362255, 0.0858973861, -0.0038466118, -0.0014717426, 0.0002783115, 0.0017762645, 0.0353550054, 0.0000361928, 0.0000930210, 0.0028033801, -3.4441223145, -0.2932579517, 0.0346499979, -0.1744651794, -0.0070120599, -0.0020394830, 0.0602010638, -0.0051756743, -0.0000147067, 0.0001341233, 0.0016865352, 0.0400307290, 0.0000128263, 0.0000956026, 0.0032877221, -3.3570098877, -0.2609835863, 0.0436842889, -0.1468238831, -0.0066699246, -0.0021913559, 0.0297321379, -0.0066978736, 0.0000869134, -0.0001095346, 0.0015123223, 0.0469819941, 0.0000023509, 0.0000105231, 0.0038720937, -3.1673507690, -0.2847481370, 0.0491102338, -0.1133394241, -0.0060390960, -0.0022864577, -0.0082556754, -0.0069544911, -0.0006451930, -0.0002582030, 0.0014278892, 0.0550367348, 0.0000518296, -0.0000461406, 0.0041305823, -3.0173873901, -0.2640793324, 0.0652342141, -0.1072034836, -0.0052130450, -0.0016828534, -0.0277243406, -0.0067722974, -0.0008306624, -0.0001864103, 0.0013585229, 0.0637045652, 0.0000671989, 0.0000305814, 0.0040879534, -2.7430267334, -0.2321515083, 0.0791397393, -0.0885105133, -0.0033249566, -0.0021803041, -0.0083223954, -0.0068406984, -0.0007689041, -0.0003636277, 0.0014062740, 0.0730689093, 0.0000504095, 0.0000404122, 0.0039370516, 0.5834238529, -0.6234794855, -5.6817054749, -1.0354195833, -3.0121254921, -2.6708705425, -1.5935447216, -3.1177000999, -2.8275914192, -6.9062500000, -6.9062500000, -2.5424056053, -6.1380968094, -5.9547100067, -4.7855558395, 0.5835173726, -0.6101937294, -5.4814043045, -1.0266287327, -3.0149641037, -2.6764752865, -1.5924346447, -3.1270680428, -2.8406956196, -6.9062495232, -6.9062495232, -2.5338561535, -6.1349902153, -5.9607653618, -4.7836127281, 0.5736096501, -0.6163725853, -5.0783972740, -1.0493633747, -3.0228202343, -2.6808109283, -1.6092368364, -3.1242179871, -2.8532233238, -6.9149904251, -6.9097695351, -2.5404992104, -6.1102614403, -5.9588131905, -4.7863306999, 0.5719485879, -0.6128364205, -4.6476202011, -1.0430619717, -3.0354394913, -2.6819381714, -1.6400229931, -3.1576435566, -2.8663742542, -6.8903317451, -6.8640756607, -2.5382313728, -6.0992584229, -5.9439449310, -4.7906827927, 0.5888072848, -0.6273163557, -4.2548108101, -1.0446046591, -3.0491905212, -2.6834092140, -1.6401213408, -3.1776876450, -2.8938500881, -6.7523264885, -6.6905760765, -2.5428392887, -6.1091089249, -5.9331493378, -4.7988162041, 0.5983048677, -0.6341117620, -3.9145140648, -1.0571227074, -3.0473699570, -2.6976671219, -1.6621094942, -3.1925742626, -2.9097080231, -6.6465139389, -6.5858259201, -2.5469985008, -6.1479549408, -5.9342179298, -4.7995519638, 0.5908662081, -0.6418309808, -3.6100404263, -1.0586986542, -3.0505638123, -2.7197363377, -1.7013735771, -3.1985621452, -2.9155602455, -6.6053619385, -6.5366430283, -2.5469372272, -6.1788148880, -5.9217114449, -4.8099241257, 0.5958960652, -0.6567019224, -3.3593144417, -1.0822504759, -3.0615460873, -2.7244040966, -1.7303705215, -3.2231638432, -2.9512805939, -6.6084365845, -6.4745311737, -2.5507013798, -6.2074847221, -5.9885601997, -4.8167963028, 0.6091979742, -0.6527036428, -3.1353144646, -1.0823609829, -3.0761382580, -2.7367033958, -1.7614686489, -3.2458708286, -2.9853415489, -6.6054840088, -6.3800673485, -2.5483961105, -6.2366013527, -6.0174393654, -4.8174724579, 0.6109117270, -0.6657885909, -2.9443767071, -1.0833457708, -3.0876960754, -2.7562699318, -1.8049550056, -3.2541866302, -3.0199980736, -6.5848751068, -6.2499980927, -2.5493319035, -6.2375540733, -6.0380249023, -4.7932515144, 0.6138614416, -0.6653077602, -2.7629117966, -1.0901244879, -3.0871293545, -2.7658486366, -1.8477697372, -3.2399673462, -3.0566391945, -6.5373759270, -6.1244096756, -2.5420498848, -6.2667684555, -6.0504446030, -4.7865409851, 0.6212841272, -0.6728774309, -2.6024568081, -1.1126110554, -3.1043000221, -2.7849566936, -1.9019374847, -3.2165775299, -3.1093375683, -6.4865436554, -5.9981155396, -2.5363974571, -6.2824912071, -6.0758857727, -4.7584981918, 0.6275119781, -0.6800500154, -2.4470469952, -1.1154370308, -3.0991785526, -2.8091351986, -1.9427205324, -3.2045669556, -3.1440851688, -6.4298782349, -5.8847427368, -2.5251288414, -6.2897920609, -6.0903730392, -4.7056365013, 0.6485821605, -0.6707342267, -2.3183641434, -1.1226527691, -3.1104664803, -2.8218116760, -1.9762033224, -3.1579914093, -3.1685500145, -6.3551683426, -5.7695980072, -2.5079212189, -6.3005042076, -6.0990123749, -4.6520638466, 0.6615297794, -0.6635304689, -2.1792674065, -1.1353344917, -3.1073498726, -2.8347015381, -2.0077354908, -3.1283547878, -3.1847858429, -6.2750434875, -5.6518511772, -2.4979999065, -6.3002624512, -6.1063089371, -4.5925579071, 0.6662154794, -0.6430724263, -2.0770525932, -1.1357837915, -3.0951604843, -2.8608870506, -2.0153424740, -3.0788290501, -3.1979804039, -6.2045040131, -5.5477728844, -2.4722158909, -6.2859349251, -6.1048035622, -4.5426735878, 0.6887686849, -0.6289377213, -1.9738515615, -1.1318747997, -3.0752286911, -2.8663692474, -1.9784754515, -3.0055947304, -3.2055220604, -6.1114606857, -5.4468884468, -2.4565162659, -6.2863998413, -6.1059741974, -4.4609994888, 0.7175059319, -0.5864943862, -1.8846776485, -1.0969853401, -3.0441863537, -2.8672924042, -1.9382971525, -2.9531273842, -3.2127251625, -6.0317168236, -5.3642134666, -2.4140803814, -6.2746996880, -6.1122694016, -4.4150066376, 0.7393589616, -0.5508284569, -1.7814462185, -1.0446758270, -3.0257811546, -2.8727445602, -1.8628975153, -2.9028716087, -3.2453033924, -5.9492974281, -5.2561349869, -2.3840980530, -6.2535257339, -6.0719232559, -4.3586339951, 0.7603103518, -0.4984786510, -1.6967229843, -0.9785599113, -2.9934339523, -2.8711771965, -1.8059529066, -2.8266439438, -3.2454967499, -5.8608922958, -5.1425065994, -2.3443245888, -6.2535018921, -6.0588579178, -4.3289165497, 0.7875124216, -0.4464786947, -1.6233546734, -0.9060804844, -2.9647798538, -2.8729813099, -1.7346737385, -2.8071656227, -3.2566134930, -5.7832708359, -5.0307202339, -2.2962017059, -6.2363872528, -6.0430574417, -4.2997150421, 0.8321235776, -0.3772444129, -1.5430189371, -0.8478833437, -2.9629487991, -2.8900363445, -1.6634662151, -2.7570755482, -3.2766261101, -5.7078423500, -4.9418225288, -2.2546558380, -6.2383556366, -6.0553345680, -4.2764387131, 0.8652449846, -0.3146486282, -1.4697751999, -0.7754581571, -2.9289593697, -2.8974683285, -1.6171064377, -2.7383656502, -3.2658345699, -5.6213612556, -4.8622674942, -2.2004408836, -6.2390165329, -6.0617351532, -4.2774677277, 0.9173734188, -0.2215024829, -1.3956376314, -0.6982303858, -2.9133090973, -2.8951342106, -1.5718258619, -2.7448282242, -3.2559790611, -5.5350933075, -4.7958445549, -2.1548948288, -6.2317914963, -6.0342621803, -4.2761282921, 0.9681003094, -0.1332409978, -1.3291317225, -0.6026791930, -2.8763184547, -2.8973939419, -1.4969164133, -2.7379477024, -3.2403328419, -5.4403319359, -4.7231354713, -2.1017327309, -6.2106952667, -6.0205984116, -4.3068218231, 1.0097792149, -0.0377216935, -1.2672849894, -0.4993644059, -2.8495466709, -2.9134874344, -1.3896987438, -2.7507057190, -3.2025074959, -5.3506050110, -4.6456413269, -2.0513031483, -6.1591720581, -5.9651932716, -4.3072724342, 1.0722999573, 0.0797096193, -1.2001370192, -0.3759214878, -2.7844851017, -2.9364714622, -1.2861589193, -2.7231321335, -3.1863527298, -5.2448525429, -4.5728683472, -2.0175857544, -6.1153388023, -5.9023852348, -4.3248090744, 1.1455378532, 0.1930467635, -1.1361086369, -0.2402370572, -2.7032024860, -2.9278674126, -1.2007631063, -2.6175358295, -3.1550035477, -5.1363086700, -4.4888920784, -1.9834309816, -6.0786190033, -5.8202524185, -4.2375917435, 1.2269661427, 0.3164764047, -1.0791907310, -0.0894319117, -2.6114292145, -2.9092822075, -1.1035065651, -2.4528837204, -3.1064717770, -5.0518317223, -4.4051871300, -1.9631645679, -5.9791188240, -5.7131094933, -4.0808091164, 1.3095047474, 0.4469480813, -1.0182459354, 0.0817531794, -2.4554021358, -2.8971791267, -1.0310416222, -2.2368535995, -3.0384559631, -4.9375615120, -4.3235378265, -1.9459738731, -5.8618636131, -5.6265192032, -3.8365232944, 1.4231333733, 0.5770505667, -0.9520523548, 0.2349257469, -2.2818698883, -2.8839130402, -0.9842026234, -1.9930914640, -2.9760856628, -4.8274154663, -4.2581367493, -1.8996834755, -5.7264223099, -5.5140724182, -3.5842065811, 1.5277776718, 0.7092797756, -0.8871574998, 0.3941540122, -2.1012332439, -2.8687067032, -0.9203348756, -1.7862931490, -2.9101762772, -4.7246251106, -4.1854596138, -1.8123825788, -5.5872612000, -5.4222288132, -3.3686797619, 1.6571321487, 0.8934124708, -0.8020926118, 0.5422254205, -1.9250277281, -2.8450021744, -0.8454668522, -1.6066874266, -2.8425993919, -4.6271419525, -4.1466498375, -1.6731464863, -5.4766530991, -5.2953920364, -3.1976189613, -14.0840311050, -0.7418906093, -0.1678199619, 0.0000518012, -0.2907381058, -0.0006696088, -0.0006637528, 0.1545892656, -0.0143313799, -0.0197000075, 0.0000621800, 0.0003476002, -0.0029982666, 0.0010719275, 0.0047175726, -0.0020182002, -0.7428685427, -0.1648528576, 0.0000414291, -0.2799978256, -0.0011558123, -0.0008724537, 0.1477531791, -0.0117476918, -0.0191243663, 0.0000788052, 0.0004040526, -0.0030064126, 0.0010378148, 0.0045104693, -0.0020224703, -0.7515828609, -0.1633273810, -0.0000277230, -0.2635192871, -0.0022027520, 0.0005392623, 0.1729383022, -0.0114036761, -0.0121604977, 0.0001203411, 0.0005680282, -0.0030496130, 0.0010460018, 0.0036589233, -0.0017369529, -0.7387259007, -0.1674789637, -0.0001140593, -0.2724924088, -0.0051559787, 0.0014869208, 0.1947516799, -0.0153977992, 0.0002849381, 0.0001640867, 0.0007478496, -0.0032032002, 0.0007777194, 0.0023695426, -0.0016023731, -0.7500152588, -0.1647931039, -0.0000873722, -0.2746973038, -0.0038575009, 0.0018761763, 0.2107623667, -0.0161392447, 0.0085796211, 0.0002133628, 0.0007926243, -0.0032593438, 0.0005064484, 0.0008244290, -0.0012744250, -0.7719802856, -0.1611314118, -0.0001011767, -0.1803340912, -0.0013078609, 0.0011269934, 0.2579782009, -0.0037202649, 0.0148404241, 0.0002509077, 0.0007091194, -0.0034727207, 0.0002527712, -0.0001665781, -0.0010282686, -0.8085670471, -0.1623224914, -0.0000097234, -0.1743879318, -0.0024800058, 0.0004240344, 0.3220305741, 0.0007938221, 0.0108891223, 0.0002282316, 0.0006381707, -0.0038578121, 0.0002178050, 0.0001423052, -0.0011386637, -0.8072810173, -0.1680524200, 0.0001586060, -0.1305208206, -0.0039761858, 0.0001741359, 0.3963775039, 0.0009308401, 0.0035858420, 0.0002592248, 0.0007943280, -0.0042561395, 0.0004056709, 0.0009505226, -0.0014920605, -0.8346033096, -0.1666083634, 0.0005595734, -0.0418930054, -0.0032856716, -0.0003816597, 0.5055717826, 0.0015008738, -0.0020499662, 0.0003480649, 0.0011147220, -0.0046601510, 0.0006703187, 0.0015872942, -0.0019119033, -0.8888759613, -0.1682265550, 0.0010707583, 0.0341482162, -0.0076874299, 0.0003190641, 0.6329050660, 0.0082596131, 0.0025821431, 0.0004603239, 0.0015619913, -0.0052342364, 0.0009111408, 0.0016662546, -0.0036714261, -0.8574695587, -0.1705898643, 0.0000469033, 0.1989097595, -0.0136620142, 0.0005398449, 0.7958418131, 0.0002930164, 0.0034094085, 0.0006542872, 0.0019493473, -0.0063310969, 0.0011101929, 0.0020877179, -0.0067204796, -0.8534030914, -0.1723939031, -0.0005740933, 0.4009819031, -0.0227740780, -0.0000812388, 0.9694486260, -0.0174691118, 0.0011673060, 0.0009890196, 0.0026087081, -0.0082304087, 0.0011608361, 0.0026206267, -0.0122567872, -0.7318630219, -0.1778811365, 0.0006384803, 0.6373462677, -0.0387446359, 0.0015646648, 1.1147736311, -0.0484078154, 0.0017248855, 0.0013125993, 0.0034315288, -0.0114845764, 0.0011490833, 0.0029140692, -0.0217673797, -0.6081676483, -0.1862172186, -0.0002540667, 0.9378538132, -0.0607832670, 0.0020838394, 1.2419493198, -0.1226291209, 0.0032314421, 0.0017235258, 0.0041698366, -0.0183223579, 0.0010121289, 0.0029836555, -0.0363457575, -0.3548240662, -0.2037047446, -0.0015602754, 1.3527021408, -0.0996249840, 0.0049782330, 1.3077193499, -0.2166984230, 0.0064865705, 0.0020523551, 0.0048865112, -0.0314044394, 0.0006321268, 0.0026587704, -0.0579052866, -0.0712013245, -0.2366201878, -0.0058605121, 1.7598237991, -0.1490795612, 0.0077360375, 1.3453865051, -0.3147037923, 0.0085656913, 0.0022965900, 0.0054712435, -0.0525147542, -0.0001126741, 0.0020099296, -0.0794432536, 0.4818744659, -0.3071525693, -0.0101211863, 2.1969842911, -0.1930182576, 0.0102834906, 1.2892467976, -0.3927275240, 0.0098837418, 0.0020666232, 0.0058131735, -0.0827992335, -0.0011459847, 0.0009602568, -0.1004521698, 1.1988792419, -0.4258608222, -0.0139280250, 2.6139030457, -0.2213943899, 0.0130182113, 1.1748389006, -0.4332655370, 0.0114505282, 0.0015733149, 0.0059885285, -0.1206931621, -0.0021047816, -0.0002327307, -0.1153577343, 1.9619674683, -0.6116111279, -0.0186116546, 3.0721435547, -0.2421203256, 0.0156723559, 1.0392812490, -0.4278982282, 0.0141295465, 0.0005993197, 0.0058845449, -0.1638818532, -0.0023165499, -0.0012760013, -0.1229259968, 3.0684280396, -0.8900382519, -0.0292328633, 3.5882120132, -0.2285488397, 0.0167659726, 0.9015375376, -0.3981754482, 0.0185957607, -0.0003785224, 0.0056437510, -0.2085096091, -0.0015119624, -0.0019528023, -0.1213116869, 4.2646026611, -1.2491762638, -0.0410149805, 3.9997320175, -0.2050311565, 0.0174387880, 0.8466010690, -0.3325591385, 0.0208802316, -0.0005016072, 0.0051410664, -0.2512583137, -0.0003327391, -0.0023687780, -0.1106843427, 5.4805984497, -1.7048091888, -0.0447454304, 4.4660425186, -0.1571753621, 0.0172840208, 0.8193072677, -0.2678598464, 0.0241785310, -0.0001350870, 0.0045323502, -0.2931381464, 0.0009834231, -0.0023098788, -0.0944104716, 7.0230178833, -2.1845626831, -0.0564536825, 4.7819070816, -0.1071680486, 0.0169471391, 0.8338450789, -0.1879253238, 0.0250418335, 0.0016202694, 0.0044476008, -0.3296066225, 0.0017006656, -0.0019206037, -0.0726896897, 8.4356231689, -2.7421519756, -0.0656534880, 5.2998499870, -0.0507213250, 0.0172868110, 0.8548054695, -0.0989661887, 0.0241067763, 0.0034644727, 0.0045058238, -0.3568969369, 0.0015280147, -0.0013302120, -0.0496580228, 9.9483222961, -3.2786836624, -0.0692631304, 5.7686777115, -0.0043833517, 0.0175012704, 0.8739882112, 0.0155360922, 0.0194039568, 0.0050162850, 0.0046618534, -0.3711549938, 0.0009050592, -0.0009028903, -0.0245736744, 11.6250686646, -3.8478116989, -0.0801175907, 6.1517634392, 0.0380548835, 0.0191622227, 0.8554427624, 0.1043439880, 0.0163638256, 0.0066684484, 0.0050471849, -0.3737605810, 0.0005954140, -0.0005921745, -0.0015844944, 13.3236846924, -4.3543000221, -0.0995691717, 6.5790500641, 0.0599606559, 0.0197147205, 0.8035386801, 0.1157382354, 0.0135120600, 0.0077958237, 0.0049257297, -0.3665001392, 0.0003215805, -0.0003746840, 0.0115074096, 14.9504089355, -4.9650740623, -0.1015016139, 6.9555015564, 0.0617282987, 0.0200207140, 0.7376554012, 0.0998369157, 0.0127266757, 0.0086612562, 0.0050963955, -0.3538594842, 0.0002469017, -0.0005573275, 0.0153569523, 17.0732269287, -5.5777559280, -0.1088664979, 7.4087858200, 0.0544371046, 0.0202962048, 0.6602379680, 0.0679479167, 0.0120690726, 0.0091867596, 0.0050373259, -0.3408067226, 0.0001022654, -0.0005839217, 0.0151972342, 18.9821853638, -6.1538619995, -0.1215093583, 7.6282711029, 0.0610074624, 0.0198384095, 0.5935319662, 0.0440818444, 0.0138130486, 0.0097759925, 0.0047700452, -0.3275245726, -0.0000914291, -0.0008042897, 0.0130338212, 21.1777496338, -6.6992568970, -0.1115628108, 8.0047225952, 0.0477869771, 0.0184199996, 0.5333437324, 0.0248391833, 0.0115005132, 0.0098963557, 0.0042332639, -0.3146412075, -0.0000457626, -0.0008004569, 0.0098830368, 22.9909667969, -7.1190328598, -0.1151857972, 8.3857402802, 0.0519496165, 0.0164819099, 0.4824259877, 0.0086395582, 0.0088414140, 0.0098228510, 0.0037014333, -0.3031203151, -0.0001492701, -0.0007033084, 0.0075433543, 25.2783355713, -7.6778821945, -0.0996938795, 8.7141246796, 0.0456667542, 0.0149786649, 0.4152596295, 0.0049195774, 0.0089540612, 0.0098536685, 0.0034866103, -0.2939278185, -0.0001823609, -0.0007930060, 0.0059457468, 0.5995818973, -0.5057460070, -5.6473746300, -0.7147982121, -2.5780711174, -2.4739952087, -1.1245129108, -2.4984443188, -2.4228711128, -6.9062500000, -6.9062500000, -2.4715664387, -5.9862380028, -5.7935895920, -4.3011784554, 0.6051247120, -0.5050702095, -5.3971862793, -0.7069405317, -2.5733885765, -2.4761331081, -1.1467468739, -2.4882435799, -2.4297220707, -6.9062500000, -6.9062495232, -2.4674773216, -5.9762816429, -5.7810454369, -4.2881979942, 0.5923797488, -0.5014171600, -4.8900337219, -0.7056438923, -2.5661952496, -2.4735209942, -1.1427290440, -2.4741299152, -2.4176411629, -6.9096031189, -6.9094719887, -2.4603750706, -5.9368185997, -5.7014698982, -4.2776947021, 0.6161901951, -0.4996056557, -4.4572196007, -0.6998106241, -2.5542232990, -2.4656822681, -1.1456097364, -2.4637413025, -2.4004268646, -6.8793306351, -6.8535847664, -2.4477052689, -5.8723425865, -5.5897994041, -4.2379708290, 0.6286824942, -0.4984042645, -4.0386652946, -0.6669592857, -2.5214047432, -2.4578506947, -1.1237955093, -2.4372797012, -2.4024944305, -6.6872706413, -6.6488895416, -2.4383578300, -5.8305678368, -5.4682741165, -4.1858224869, 0.6347694397, -0.4994800091, -3.6956937313, -0.6275414228, -2.4908680916, -2.4547462463, -1.0877021551, -2.3747081757, -2.4104797840, -6.5368289948, -6.5117244720, -2.4162023067, -5.8111333847, -5.3828935623, -4.0976400375, 0.6629149914, -0.5091730356, -3.3803319931, -0.5826533437, -2.4221248627, -2.4486744404, -1.0321362019, -2.2539341450, -2.3573219776, -6.4675626755, -6.4018115997, -2.4016385078, -5.7766056061, -5.2744784355, -4.0015816689, 0.6796795130, -0.4753276110, -3.1099531651, -0.5252373219, -2.3611743450, -2.4448196888, -0.9535915852, -2.1363062859, -2.3276202679, -6.4444642067, -6.2240400314, -2.3666992188, -5.7109031677, -5.2555875778, -3.8688211441, 0.7011384964, -0.4758853912, -2.8761682510, -0.4600089788, -2.2645463943, -2.4266767502, -0.8715329170, -1.9964082241, -2.3205616474, -6.3918628693, -5.9721670151, -2.3283486366, -5.6068425179, -5.2018246651, -3.6964707375, 0.7209799290, -0.4581695795, -2.6569590569, -0.3707843721, -2.1521091461, -2.4128580093, -0.8053858280, -1.8138285875, -2.2887489796, -6.2784566879, -5.7128729820, -2.2949681282, -5.4885320663, -5.1353049278, -3.4966435432, 0.7564833164, -0.4218189716, -2.4703998566, -0.2928824723, -1.9903074503, -2.3920192719, -0.7470961809, -1.6055560112, -2.2587127686, -6.0965113640, -5.4615454674, -2.2400579453, -5.3722248077, -5.0278692245, -3.2454857826, 0.8022087216, -0.3933289051, -2.2767837048, -0.1845408976, -1.8303017616, -2.3650460243, -0.6977608204, -1.3713526726, -2.2110664845, -5.8971300125, -5.2233071327, -2.1647751331, -5.2094154358, -4.9183173180, -2.9894943237, 0.8421856761, -0.3250550032, -2.1005768776, -0.1027154624, -1.6207134724, -2.3323435783, -0.6710437536, -1.1150927544, -2.1651415825, -5.6885094643, -5.0139040947, -2.0585672855, -5.0197901726, -4.7847084999, -2.7147512436, 0.8919885159, -0.2332367897, -1.9360933304, -0.0217093527, -1.4292776585, -2.2938475609, -0.6656317711, -0.8778109550, -2.0777142048, -5.4642834663, -4.8084211349, -1.9048967361, -4.8398008347, -4.6763715744, -2.4313337803, 0.9671470523, -0.1143083572, -1.7944133282, 0.0323503017, -1.2694579363, -2.2570738792, -0.7031502128, -0.6940248013, -2.0068192482, -5.2503585815, -4.6057362556, -1.7270064354, -4.6608209610, -4.5377225876, -2.2241508961, 1.0313223600, 0.0564839840, -1.6325490475, 0.0847579837, -1.1273512840, -2.2120792866, -0.7117868662, -0.5242421627, -1.8746657372, -5.0294766426, -4.4109344482, -1.5377495289, -4.5001273155, -4.3768138885, -2.0457222462, 1.1122741699, 0.2523872852, -1.4943280220, 0.1328682899, -1.0110538006, -2.1813845634, -0.7711808085, -0.4303884506, -1.7937046289, -4.8073329926, -4.1952886581, -1.3467772007, -4.3165884018, -4.2352519035, -1.9194962978, 1.1954207420, 0.4763422012, -1.3563008308, 0.1697590351, -0.9472799301, -2.1638088226, -0.7980082631, -0.3415406942, -1.7187376022, -4.5984382629, -4.0015535355, -1.1913759708, -4.1750650406, -4.1383938789, -1.8245575428, 1.2676239014, 0.7027567625, -1.2194654942, 0.2029170990, -0.9133352041, -2.1279566288, -0.8324715495, -0.2965815067, -1.6559793949, -4.3992934227, -3.8172729015, -1.0375645161, -4.0607051849, -4.0513467789, -1.7959394455, 1.3494583368, 0.9131089449, -1.0839681625, 0.2352771759, -0.8692922592, -2.1162848473, -0.8395841122, -0.2550313473, -1.6274982691, -4.2196283340, -3.6657686234, -0.9300239086, -3.9951119423, -3.9729287624, -1.8051743507, 1.4322125912, 1.1277868748, -0.9411573410, 0.2823227644, -0.8473218679, -2.0757169724, -0.8173156381, -0.2209243774, -1.6049191952, -4.0573859215, -3.5446922779, -0.8146519661, -3.9668302536, -3.9393515587, -1.8179547787, 1.5099873543, 1.3272881508, -0.7988731861, 0.3582541347, -0.8227901459, -2.0607221127, -0.7666924000, -0.2331883907, -1.5769550800, -3.9362573624, -3.4706256390, -0.7406635284, -3.9831500053, -3.9032692909, -1.8551445007, 1.5827935934, 1.5207624435, -0.6608538628, 0.4308737516, -0.7955894470, -2.0466616154, -0.7267214656, -0.2561357021, -1.5797772408, -3.8381662369, -3.4112970829, -0.6642050743, -4.0006971359, -3.8826179504, -1.8953666687, 1.6817276478, 1.7185652256, -0.5096099377, 0.5162611008, -0.7718191147, -2.0305559635, -0.6924976110, -0.2675583363, -1.5903835297, -3.7416753769, -3.3940045834, -0.6247198582, -4.0605945587, -3.9098577499, -1.9058423042, 1.7811348438, 1.9105889797, -0.3814218044, 0.5680887699, -0.7374534607, -2.0489354134, -0.7041785717, -0.3233261108, -1.6391309500, -3.6757702827, -3.3689789772, -0.5884087086, -4.1386613846, -3.9711368084, -1.9588308334, 1.8815075159, 2.0999798775, -0.2307374477, 0.6397588253, -0.7033674717, -2.0605669022, -0.7670457959, -0.3996686935, -1.6596399546, -3.6033422947, -3.3617084026, -0.5791757107, -4.2157554626, -4.0431571007, -2.0268292427, 1.9699765444, 2.2881448269, -0.1021629572, 0.7029328346, -0.7086944580, -2.1086294651, -0.7960207462, -0.5931706429, -1.7031226158, -3.5648045540, -3.3372130394, -0.5504190922, -4.3234066963, -4.1468248367, -2.2167201042, 2.0756144524, 2.4591724873, 0.0386660099, 0.7459223866, -0.6796875000, -2.1475763321, -0.8410722613, -0.7551911473, -1.7730109692, -3.5479793549, -3.3185255527, -0.5193476677, -4.4022145271, -4.2483263016, -2.5026955605, 2.1902139187, 2.6256649494, 0.1659194231, 0.7832930088, -0.6468451023, -2.1854152679, -0.8814282417, -0.8945206404, -1.7837756872, -3.5225872993, -3.2901029587, -0.4905142784, -4.4911899567, -4.2988247871, -2.7287626266, 2.2730612755, 2.7965950966, 0.3176776171, 0.8041039109, -0.6039351225, -2.2230546474, -0.9061778188, -0.9678478837, -1.7773097754, -3.5063254833, -3.2536010742, -0.4743521214, -4.5352473259, -4.3589386940, -2.8770616055, 2.3847200871, 2.9451425076, 0.4379627705, 0.8125592470, -0.5726243258, -2.2613148689, -0.9275960922, -1.0067209005, -1.7793492079, -3.4876155853, -3.2288160324, -0.4452235699, -4.5721349716, -4.3952236176, -2.9871213436, 2.4788503647, 3.0957295895, 0.5689829588, 0.8625880480, -0.5522037745, -2.2813062668, -0.9101989269, -1.0371477604, -1.7861015797, -3.4752573967, -3.2016849518, -0.4500286579, -4.5800132751, -4.4318804741, -3.0838403702, 2.5729358196, 3.2365875244, 0.7037671804, 0.8928023577, -0.5336222649, -2.3136956692, -0.9054031372, -1.0505543947, -1.7900249958, -3.4642105103, -3.1731834412, -0.4443302155, -4.5776143074, -4.4439482689, -3.1205477715, -16.1522159576, -4.0687265396, 1.2317272425, -4.0190296173, 1.2312240601, -4.0290789604, 1.2252373695, -4.0403804779, 1.2188673019, -4.0022282600, 1.2110303640, -3.9799625874, 1.2036626339, -4.0203790665, 1.1799247265, -3.9610919952, 1.1632373333, -3.9999499321, 1.1571012735, -4.0515332222, 1.1551686525, -4.2516851425, 1.1495709419, -4.3165612221, 1.1346964836, -4.3943929672, 1.1165618896, -4.5121273994, 1.1226406097, -4.6687989235, 1.1701358557, -5.2334651947, 1.1628904343, -5.1114096642, 1.1469970942, -4.8968806267, 1.1548656225, -4.4535803795, 1.0790394545, -4.1931619644, 1.1016242504, -4.5033197403, 1.1288827658, -3.9387869835, 1.1987254620, -3.7219889164, 1.0137525797, -3.0171623230, 0.8961003423, -3.7435548306, 0.9311795235, -4.1249084473, 0.9069975615, -2.7486982346, 0.8422508240, -3.1695847511, 0.6562221050, -5.0902910233, 0.5889514685, -1.7790050507, 0.5493938923, -2.8791587353, 0.1836184263, -1.4823112488, -0.0977178812, -1.0318136215, -0.4358503819, -1.9224025011, 1.2251999378, -1.9405621290, 1.2195734978, -1.9326918125, 1.2135179043, -1.9095861912, 1.2051242590, -1.9100661278, 1.1988641024, -1.9222252369, 1.1804846525, -1.9191608429, 1.1697359085, -1.9544752836, 1.1576209068, -2.0531857014, 1.1448756456, -2.2032546997, 1.1327974796, -2.4755835533, 1.1157209873, -2.7251296043, 1.1040772200, -3.0091075897, 1.0990221500, -3.3639702797, 1.1160081625, -3.5755453110, 1.1219794750, -3.7322239876, 1.1504468918, -3.9066889286, 1.1144500971, -3.8264119625, 1.0799504519, -3.9064328671, 1.1053749323, -3.9459061623, 1.0649737120, -3.8479404449, 1.0795718431, -3.8108565807, 1.0290484428, -3.0129308701, 0.9914277196, -3.0713891983, 0.9418867230, -2.6667308807, 0.9044427872, -3.7603757381, 0.7681876421, -2.6364445686, 0.7729169130, -2.7007145882, 0.8174884915, -2.8940563202, 0.3980592489, -2.7837419510, 0.4184664488, -3.3077726364, 0.0717273951, -3.8993387222, -0.0321533680, -3.6885974407, -0.2662560940, 1.3453261852, 1.2315740585, 1.3542716503, 1.2287590504, 1.3672473431, 1.2281506062, 1.3818783760, 1.2179967165, 1.4384381771, 1.2045735121, 1.4726186991, 1.1916332245, 1.5328512192, 1.1780622005, 1.5137754679, 1.1639163494, 1.4571976662, 1.1418472528, 1.3078503609, 1.1296967268, 1.1165335178, 1.1174770594, 0.8729846478, 1.0966291428, 0.6446325779, 1.0913243294, 0.3642997742, 1.1031857729, 0.1499947309, 1.1017565727, 0.0578922033, 1.0958858728, -0.0000274181, 1.0796828270, 0.3175334930, 1.1083437204, 0.3188786507, 1.0912199020, 0.1390534639, 1.0013346672, 0.8367509842, 0.9743666053, 0.9401037693, 1.0274572372, 1.2263264656, 0.8013259768, 1.1188719273, 0.7722558379, 1.4327554703, 0.6904218197, 2.2580244541, 0.5640251040, 2.0805196762, 0.5158729553, 3.7941093445, 0.5615228415, 2.5489978790, 0.2940346003, 3.7232778072, 0.0873031616, 2.3693895340, -0.1035678387, 1.7435266972, -0.2846645117, 3.8371229172, -0.5114769936, 4.5065274239, 1.2640887499, 4.4652318954, 1.2628493309, 4.5081934929, 1.2563743591, 4.5885224342, 1.2518472672, 4.6652679443, 1.2372946739, 4.6431159973, 1.2271698713, 4.7024359703, 1.2088831663, 4.6988124847, 1.1901016235, 4.6166877747, 1.1779725552, 4.5251555443, 1.1712185144, 4.1224546432, 1.1674635410, 3.6575350761, 1.1402777433, 3.4246978760, 1.1278823614, 3.2193338871, 1.1374745369, 3.0026440620, 1.1630718708, 3.0203185081, 1.1564557552, 3.3254463673, 1.1147644520, 3.1721730232, 1.1456701756, 3.6361589432, 1.1395937204, 3.6657977104, 1.0839830637, 3.9078979492, 0.9846413136, 4.5635900497, 0.9063242078, 4.8279399872, 0.9636899829, 4.3465352058, 0.8320039511, 4.0821390152, 0.8583642840, 5.9972887039, 0.8632994890, 5.3606195450, 0.5933305025, 6.7904028893, 0.4982848763, 5.5647811890, 0.5991621017, 5.3952884674, 0.2172349095, 7.3470964432, 0.3888665438, 10.2301445007, 0.0775728226, 7.4190063477, -0.0329506397, 0.5821602345, -1.2947703600, 0.6634250879, -1.2916669846, 0.6314837933, -1.3118007183, 0.6678105593, -1.3382017612, 0.7344944477, -1.3794922829, 0.8155269623, -1.4317588806, 0.8845026493, -1.4584472179, 1.0025354624, -1.4828077555, 1.1329224110, -1.4117715359, 1.3138985634, -1.1592960358, 1.5633559227, -0.8731591702, 1.7286446095, -0.5273624659, 1.9130907059, -0.2209463120, 2.1987714767, 0.0242562294, 2.3273773193, 0.2383108139, 2.4731860161, 0.4611535072, 2.6416788101, 0.6501526833, 2.6806943417, 0.8563678265, 2.8768630028, 1.0338916779, 3.0633733273, 1.2256171703, 3.1288897991, 1.2837533951, 3.3181924820, 1.4389023781, 3.5990028381, 1.6490726471, 3.8888075352, 1.7544448376, 4.0237035751, 2.0004053116, 4.0680031776, 2.1553187370, 4.2223663330, 2.2324147224, 4.1978349686, 2.3769853115, 4.3239469528, 2.4533522129, 4.2383842468, 2.6078453064, 4.2876944542, 2.7730870247, 4.3251576424, 2.8081512451, 4.4232988358, 2.8498625755, 0.1122368574, -1.3284999132, 0.1364376545, -1.3377029896, 0.1397418976, -1.3650823832, 0.1703795195, -1.4334055185, 0.2976579666, -1.4787731171, 0.4065604210, -1.5954251289, 0.4557042122, -1.6133639812, 0.7275602818, -1.4470353127, 0.9579482079, -1.2200354338, 1.1768445969, -0.9451588392, 1.4908425808, -0.6049060822, 1.7627203465, -0.2798058987, 2.1224176884, -0.0890803337, 2.3550317287, 0.1570236683, 2.4901456833, 0.3252060413, 2.7355844975, 0.5535564423, 2.9814217091, 0.7004408836, 3.2379603386, 0.9576144218, 3.4574422836, 1.1480109692, 3.6326513290, 1.3949322701, 3.7946271896, 1.5649671555, 4.1446595192, 1.6972417831, 4.2604913712, 1.8725123405, 4.4147987366, 1.9912879467, 4.5572257042, 2.0778009892, 4.6201691628, 2.2235724926, 4.6964454651, 2.2890338898, 4.7348833084, 2.4847869873, 4.8724441528, 2.5707349777, 4.8394184113, 2.5818452835, 4.9356698990, 2.6980066299, 4.8774318695, 2.8085951805, 4.9127612114, 2.8642897606, 0.1635153294, -1.2837260962, 0.1531748772, -1.2909870148, 0.1544289589, -1.3182977438, 0.2357275486, -1.3795605898, 0.3013088703, -1.4397845268, 0.4182276726, -1.5476053953, 0.5139374733, -1.5728490353, 0.7083373070, -1.4351758957, 0.9057726860, -1.2269322872, 1.1037209034, -0.9306738377, 1.3192842007, -0.6501622200, 1.5462207794, -0.3339333534, 1.8324127197, -0.0777976513, 2.0616650581, 0.0985877514, 2.3392918110, 0.3149127960, 2.4839849472, 0.5048661232, 2.7060217857, 0.7021148205, 2.7831437588, 0.9389286041, 3.1003437042, 1.0750343800, 3.3786799908, 1.3214819431, 3.5892047882, 1.5271716118, 3.9415416718, 1.5986046791, 4.1259231567, 1.8513658047, 4.3749814034, 1.9369928837, 4.5674495697, 2.0373203754, 4.7259893417, 2.2846736908, 4.9282016754, 2.3199319839, 4.9890375137, 2.4300031662, 4.9965353012, 2.5580010414, 4.8614883423, 2.6338396072, 4.9330124855, 2.6594924927, 5.0030069351, 2.8052935600, 4.9672002792, 2.9120807648, 0.6137147546, -1.1570420265, 0.6482126713, -1.1650589705, 0.6275276542, -1.1937952042, 0.7056033611, -1.1874290705, 0.6965022087, -1.2308748960, 0.7738177776, -1.2723186016, 0.8276469707, -1.2845685482, 0.8897416592, -1.2959743738, 1.0221014023, -1.2412900925, 1.1150047779, -1.0034474134, 1.2740150690, -0.7211866379, 1.3401834965, -0.5524914265, 1.4986031055, -0.2246704102, 1.6780639887, 0.0244560242, 1.8947567940, 0.3213980198, 2.0976657867, 0.5012435913, 2.2546646595, 0.6578269005, 2.4402852058, 0.7873375416, 2.6669340134, 0.9317235947, 2.8798670769, 1.1973938942, 3.0715343952, 1.3712694645, 3.2220921516, 1.5473017693, 3.4563624859, 1.6381368637, 3.5909645557, 1.7708368301, 3.8809118271, 1.9197516441, 4.0222878456, 2.0691094398, 3.9816079140, 2.1732091904, 4.1776485443, 2.2382888794, 4.4239873886, 2.2971463203, 4.4386434555, 2.4126324654, 4.6910610199, 2.5306801796, 4.5611734390, 2.6464867592, 4.6741447449, 2.7657213211, -1.1066789627, -5.0850167274, 0.8306148648, -3.5553126335, -0.1917318106, -4.0858249664, -2.3364696503, -6.4579048157, -2.9558460712, 1.2283788919, -2.9160261154, 1.2279996872, -2.9482598305, 1.2280299664, -3.0003750324, 1.2219057083, -3.0021495819, 1.2177180052, -2.9818758965, 1.2085150480, -3.0349338055, 1.2087383270, -3.0334362984, 1.2055970430, -3.1424212456, 1.2101792097, -3.1217420101, 1.2049725056, -3.2110261917, 1.2037392855, -3.1523678303, 1.2075527906, -2.9226660728, 1.2026032209, -2.5846443176, 1.1939791441, -2.1048271656, 1.1669616699, -1.5373144150, 1.1364507675, -1.0676648617, 1.1064025164, -0.6631379128, 1.0832608938, -0.4015707970, 1.0460211039, -0.2592654228, 0.9759453535, -0.4376769066, 0.9534711242, -0.7763757706, 0.8543140888, -0.8674106598, 0.7691438794, -1.7852759361, 0.6501475573, -1.7091169357, 0.4898061752, -2.2846724987, 0.3323352337, -3.2505939007, 0.1869418025, -3.2579874992, -0.0258342028, -3.2905881405, -0.2700141668, -3.7292222977, -0.5202249289, -3.7823021412, -0.7725350857, -2.9056987762, -1.0757124424, -2.6730659008, -1.2957038879, 2.5930421352, 1.2458477020, 2.6123836040, 1.2436795235, 2.6590466499, 1.2381943464, 2.7160775661, 1.2351734638, 2.8124313354, 1.2307903767, 2.9632623196, 1.2230118513, 3.1758811474, 1.2161637545, 3.4524765015, 1.2117236853, 3.8771257401, 1.2051702738, 4.4522681236, 1.1951301098, 5.1357545853, 1.1912791729, 5.9340639114, 1.1760149002, 6.7347736359, 1.1636151075, 7.5377645493, 1.1327729225, 8.1809768677, 1.0825853348, 8.7230873108, 1.0385272503, 9.1763858795, 0.9946329594, 9.4764127731, 0.9601561427, 9.5819358826, 0.9062085152, 9.4291620255, 0.8783354759, 9.2513294220, 0.8256734610, 8.8339481354, 0.7673494816, 8.3426284790, 0.7186462283, 7.8753523827, 0.6242724657, 7.2551975250, 0.5360307693, 6.7598791122, 0.4066067934, 6.1047182083, 0.2552559376, 5.5906543732, 0.0808314085, 5.3659925461, -0.1440213919, 5.1084961891, -0.2996834517, 4.7536067963, -0.5516521931, 4.7681903839, -0.8218977451, 6.3357658386, -1.0714519024, 0.8591495752, -1.0656520128, 0.8528913260, -1.0520056486, 0.8557831049, -1.0438534021, 0.8745940328, -1.0413703918, 0.9291862845, -1.0382257700, 0.9861353040, -1.0443747044, 1.0714573860, -1.0329016447, 1.1638419628, -1.0034857988, 1.3374530077, -0.9273824692, 1.5307575464, -0.8345502615, 1.7760570049, -0.6924734116, 2.1484642029, -0.4366555214, 2.4670817852, -0.1666848660, 2.8022677898, 0.1203379631, 3.0800249577, 0.3989551067, 3.3378491402, 0.6438117027, 3.5661036968, 0.8537330627, 3.7718100548, 1.0041639805, 3.9165444374, 1.1744253635, 4.0831189156, 1.3533644676, 4.2128620148, 1.4553296566, 4.3664913177, 1.6314883232, 4.4839491844, 1.7011680603, 4.5686616898, 1.8574278355, 4.7514653206, 1.9157402515, 4.8219237328, 2.0261182785, 4.8926248550, 2.0873737335, 5.0297679901, 2.2069616318, 5.1562042236, 2.2564382553, 5.2641534805, 2.3494436741, 5.3122596741, 2.4336171150, 5.4945197105, 2.5427415371, 5.5629792213, 2.6359713078, 0.9840811491, -0.8354938030, 1.0120562315, -0.8231389523, 1.0117064714, -0.8109018803, 1.0476770401, -0.8169888258, 1.0941979885, -0.7912454605, 1.1597670317, -0.7882471085, 1.2817009687, -0.7462263107, 1.4373539686, -0.7240698338, 1.6081589460, -0.6456606388, 1.8505239487, -0.5448421240, 2.1165456772, -0.3396056890, 2.4528913498, -0.0710971355, 2.7499976158, 0.1664214134, 3.0281527042, 0.4147734642, 3.3308458328, 0.6444616318, 3.5806660652, 0.8470189571, 3.7181174755, 1.0128972530, 3.8875398636, 1.1631264687, 4.0280733109, 1.3012278080, 4.1372046471, 1.4082889557, 4.2231760025, 1.5899770260, 4.3339519501, 1.6957247257, 4.4269685745, 1.8277542591, 4.6135411263, 1.9936587811, 4.6793470383, 2.0390248299, 4.7684888840, 2.1677181721, 4.8556528091, 2.2494263649, 4.9851045609, 2.3291838169, 5.0613718033, 2.4408643246, 5.1749935150, 2.5215604305, 5.2191925049, 2.5996050835, 5.4424047470, 2.6403720379, 5.5224394798, 2.7170503139, 3.6598625183, 0.3759893477, -0.1118335724, -0.0342918634, -0.0388100892, 0.2717647552, -0.2702178955, -0.0125317601, -0.6651237011, -0.7252837420, -0.3002414703, -0.0085382164, -0.0962539688, 0.2957684398, -0.3381175995, -0.0087326262, -0.5095914006, 0.0586099178, -0.3342370987, -0.0169491954, 0.4261736870, -0.1309283823, -0.3368501663, -0.0019367393, 1.0021741390, -0.1918017119, -0.8695410490, -1.3115448952, 0.5484898090, 0.5699470043, -1.4561266899, -1.9277184010, -1.1925849915, -0.7996026278, -1.5420181751, -2.0374484062, -1.2756195068, 0.7640361190, -1.2688049078, -1.7974226475, 0.2165568918, 0.0248164684, -0.7416239977, -1.2957080603, -0.7476561666, 0.1584482640, -0.1562004685, -0.9534961581, 2.5874364376, 0.8861557245, -2.4674592018, 3.7571487427, 0.4081672430, 0.3496303558, -0.0096913166, -0.2487554252, -0.0088014565, 0.7222509384, 0.2674250305, 0.6458575130, 0.1976907849, 2.0615577698, 0.6712169647, 0.3221523166, -0.3878743052, 3.6033067703, 0.8542060256, 1.1713539362, -1.0199980736, 5.0345106125, 0.7736996412, 0.3088869154, 0.8984020948, 6.4856405258, 0.5641310215, 1.2967399359, -0.1228018701, -0.3929994106, -1.0323972702, 0.2022190988, 1.2237362862, 0.1788777262, -0.5020657778, -0.6833362579, 0.1248462945, 0.5809764862, -0.3422169089, -0.2421735525, -0.1151779592, 0.6854314804, -0.3767969608, -0.4435591400, 0.0450821891, 0.7087950706, -0.4106830359, -0.7911114097, 0.0467234254, 0.7979345918, -0.5372457504, 1.8607913256, -0.0091180503, -3.5736267567, -0.5763755441, -0.9437656403, -1.6121392250, 3.0145044327, 1.9568934441, 1.9998216629, -6.9560775757, -7.5954747200, -3.3793935776, -3.3875851631, -17.6754455566, -5.0313901901, -2.8369057178, -1.7543027401, 0.0441058874, -5.4394373894, -7.3974308968, -8.1250734329, -0.9899089336, -2.6772372723, -1.3860050440, 0.6850059032, -4.9848165512, -6.9769001007, -7.7743258476, -0.4693052769, -2.4746489525, -1.2150509357, 1.0212153196, -4.6352915764, -6.6060199738, -7.4639406204, -0.1119379997, -2.2830295563, -1.1080579758, 1.2458899021, -4.4212121964, -6.3989968300, -7.2200717926, 0.2090638876, -2.1048030853, -1.0331985950, 1.4208384752, -4.2766270638, -6.2344632149, -7.0571279526, 0.5415829420, -2.0098428726, -1.4110560417, -1.7355206013, -1.2407107353, -1.5667442083, -1.1725051403, -1.4592305422, -1.1381208897, -1.3845860958, -1.1080011129, -1.3338663578, -1.0788173676, 2.3409309387, 1.5780994892, 2.0379920006, -4.0814208984, -3.9125127792, -3.2911775112, -1.3428682089, -12.8846340179, 1.7760919333, 0.9852352142, 1.4059369564, -5.5026798248, -4.5119833946, -4.0767755508, -1.9927575588, -13.6924896240, 2.1212348938, 1.2781771421, 1.6444228888, -5.0938968658, -3.8524305820, -3.4439558983, -1.8220832348, -13.7191972733, 2.8739864826, 2.0073168278, 2.2505998611, -4.0222892761, -2.8355894089, -2.5721840858, -1.1742869616, -12.5233879089, 2.2168865204, 0.0376134180, -0.0025179982, -0.0023929866, 0.0085732117, 0.0223701522, 0.3296144009, -2.0793440342, -1.4879803658, -3.5818111897, -3.9633860588, -2.1193532944, -0.0014534668, 0.0569845811, -0.0111466832, -3.5879478455, -3.1379568577, -3.1609327793, 0.0580844879, -0.0001650769, 0.0089581152, 0.0000870947, 0.0007594777, 0.0010390468, -0.0653385520, -4.4496483803, -2.3798174858, -5.0998277664, -5.0720319748, -5.0853648186, 0.0000003025, 0.0000000242, 1.1586962938, 0.0015192416, -0.0006332677, -0.0008082265, -6.9202036858, -6.9204845428, -3.0462594032, -4.6766071320, -4.2800359726, -4.5929136276, 0.0008035941, -5.6199092865, -0.0289517343, -0.0477200709, -0.0009180292, -0.0103810076, -0.0177165829, -0.0506878532, 0.0257429611, -0.0375427268, -0.0371994898, -0.0326896086, 0.0204655230, -0.0298941005, -0.0262682308, 0.0096322866, -0.0296365712, 0.0105595589, -0.0211757496, 0.0009030963, 0.0023045505, 0.0195266660, 0.0023639896, -0.0123805236, -0.0050518517, 0.1160515323, -0.0057066176, 0.0375247672, 0.0371377096, -0.0013830111, -0.0025142252, 0.0471801013, -0.0212426763, 0.0112847621, 0.0021574446, 0.0033953977, 0.0472436957, -0.0313455611, -0.0194504950, 0.0084390594, 0.0227780584, 0.0291281808, 0.0437938310, 0.0600342527, -0.0292616244, 0.1099599749, 0.0047677904, -0.0324929804, 0.0546726398, 0.0411066711, 0.0237924717, -0.0002805219, 0.0455899909, -0.0736126974, -0.0199121535, -0.0065354882, 0.1341488063, -0.0519237034, -0.0319166072, -0.0034272508, -0.0121171428, -0.0027297505, 0.1587117612, 0.0017945212, 0.0133922743, -0.0395074189, -0.0786382258, -0.0367876291, 0.0344692692, 0.1887272000, 0.0610853322, 0.0583214164, 0.0182576198, 0.0215122588, -0.0411637686, -0.0566559508, 0.0538558811, -0.0397347771, 0.1017381325, 0.0104217259, 0.0085830931, 0.0195461474, -0.0943338796, -0.0136622079, 0.0473622791, 0.0232039317, -0.0751722157, -0.0343429036, 0.0197520200, 0.0400296301, 0.0441811495, -0.0394455642, -0.0368167609, 0.0131253842, -0.0392527096, 0.0180518050, -0.0277438816, 0.0167566650, -0.0000133094, -0.0185295418, -0.0651602447, -0.0998218134, 0.0109896148, -0.0088751400, -0.0045817215, 0.0806122348, 0.0748260468, -0.0034588149, -0.0082779275, -0.0269137248, 0.0089090718, 0.0036557361, -0.0556729846, 0.0395250991, -0.0020645126, 0.0017805976, 0.1020999178, -0.0061657489, -0.0114406142, -0.0317786597, -0.0016501358, 0.0190916378, 0.0285340231, -0.0033801869, -0.0227780286, 0.0138158360, 0.0619422831, -0.0313271917, 0.0499926917, -0.0068265991, -0.0110968510, 0.0014564910, 0.0162293203, 0.0941636339, 0.0162199009, 0.0162017066, 0.0425690114, 0.0636823997, 0.0279541425, 0.0124448771, 0.0295550171, -0.0192070063, -0.0099207060, 0.0241491068, 0.0276003163, -0.0055668503, 0.0958075076, -0.0365276411, 0.0217334442, 0.0283689275, 0.0607438497, 0.0239896271, 0.0385743082, -0.0085077416, 0.0648147836, -0.0068270671, -0.1136957482, -0.0287356842, 0.0191476829, 0.0017167123, -0.0217617154, -0.0106607126, -0.0171830356, -0.0247353204, -0.0457547344, -0.0082648369, 0.0218638610, -0.0259903129, 0.0237672869, 0.1029657945, 0.0417945497, -0.0485410579, 0.0194779914, -0.0202009138, -0.0457348451, 0.0183715820, 0.0129854446, -0.0680509359, 0.0624355823, 0.0012581657, -0.0213683266, 0.0089923516, -0.0409299918, 0.0068388260, -0.0388598032, 0.0217888597, -0.0358261913, 0.0060829087, -0.0050705140, -0.0346862003, -0.0304725785, 0.0268205870, -0.0257375799, 0.0206342451, -0.0457252488, 0.1566023529, -0.0146138761, 0.0196317956, 0.0003341692, -0.0190850850, 0.0015160469, 0.0181629602, 0.0518990718, -0.0040537906, 0.0193639994, 0.0283959564, 0.0432372354, 0.0473485924, -0.0695642903, -0.0044340566, 0.0829367936, 0.0302799866, -0.0743229240, -0.0536352061, -0.0472153500, -0.0486118458, 0.0030234356, 0.0342506431, -0.0156722534, 0.0052727591, -0.0191749465, 0.0097625144, 0.0398456976, -0.0092682811, 0.0522146374, -0.0475671589, 0.0082422551, -0.0549863912, 0.0271283127, -0.0110815084, 0.0077513270, 0.0931665674, -0.0027745636, 0.0172871705, 0.0646825656, 0.0650801510, -0.0395485610, 0.0348296501, -0.0237194598, 0.0174491070, -0.0101959044, -0.0473497026, -0.0112295356, 0.0964363590, -0.0517615899, 0.0537456013, 0.0065092789, -0.0021358279, 0.0282337852, 0.0246367864, -0.0090773264, -0.0046552015, 0.0370227173, -0.0152798155, -0.0057474524, -0.0219374578, -0.0422002077, -0.0308418497, 0.0012697492, 0.0198556874, 0.0610317439, -0.0055464362, -0.0366406702, -0.0468657054, -0.0361434519, 0.0120279845, 0.0547356494, -0.0328104980, 0.0135915363, -0.0721899047, 0.0046667410, 0.0189576652, 0.0152594568, 0.0214208737, -0.0072960858, -0.0073437658, -0.0207771100, -0.0755154490, 0.0745922700, 0.0499213189, -0.0062336712, -0.0297042746, -0.0110774273, -0.0205467250, -0.0242491197, 0.0129017858, 0.0051828986, -0.0208058339, 0.0131613631, -0.0294740498, 0.0393779054, 0.0082805157, 0.0195280705, 0.0185690895, -0.1419168264, -0.0377870947, -0.0160363186, 0.0071918918, 0.0063939695, 0.0186923556, 0.0849379450, -0.0193893015, 0.0489532128, 0.0047027059, 0.0201496053, 0.0601109043, -0.0288292300, -0.0328042768, 0.0244961455, -0.0190148819, 0.0819180161, -0.0658793896, -0.0412785523, 0.0133897010, 0.0388612077, 0.0138221551, -0.0421786793, 0.0344776958, 0.0253061783, 0.0652271137, -0.1133285165, -0.0107560745, 0.0127091939, -0.0127054937, 0.0391157269, 0.0063586626, -0.0190885011, -0.0326313116, 0.0068567866, -0.0187799335, 0.0244154111, 0.0209704619, 0.0026079763, -0.0153273204, -0.0131283095, 0.0443476215, -0.0115943607, -0.0512460619, 0.0235851370, 0.1273577809, -0.0047258143, -0.0565737560, -0.0868549943, -0.0283557065, -0.0202790145, -0.0253231451, -0.0112380181, 0.1356593221, 0.0340285078, 0.0173489507, -0.0056464765, 0.0419705249, 0.0085361740, -0.0086067868, -0.0890123248, -0.0186512265, -0.0053497492, -0.0040085092, -0.0189963952, 0.0541734360, 0.0186168272, 0.0221956912, 0.0290612523, 0.1460176557, 0.0581887588, 0.0481837504, 0.0125642233, 0.0329563841, 0.0020763304, -0.0200169217, -0.0190509204, -0.0252850000, 0.0317083970, 0.0439648367, 0.0109742284, -0.0224514659, 0.0291858632, -0.0744874328, 0.0120942099, 0.0285574254, 0.0233197678, 0.0138499439, 0.0424225219, 0.0916922316, 0.0337270424, 0.0221409313, 0.0373665728, 0.0141449748, -0.0087969806, 0.1027592495, 0.0171254687, 0.0146582210, 0.0302544795, -0.0062362161, 0.0085586244, 0.0030175853, 0.0134236319, -0.0531805232, 0.0264427755, 0.0189666748, 0.0331635438, 0.0413890071, 0.0431880914, -0.0477224253, -0.0046588285, 0.0088756084, -0.0015070376, -0.0454282872, 0.0423479863, -0.0280461088, -0.0256744102, 0.0331873521, 0.0330116116, 0.0224877447, 0.1020273194, 0.0062566628, 0.0335097052, -0.0395487361, -0.1234598383, -0.0220439341, 0.0547236539, 0.1870140433, 0.0127810352, 0.0109386584, -0.0210039839, 0.0294979196, 0.1699663401, -0.0160611812, 0.0143585103, 0.0217488445, -0.0110914689, 0.0409020260, 0.0399555676, -0.0304530393, -0.0172462184, -0.0438337885, 0.0364089981, -0.0349883102, 0.0267985910, 0.0201185998, -0.0303206835, -0.0058304099, 0.0019082511, 0.0460492410, -0.0565185882, 0.0034727387, -0.0546970367, -0.0077608335, -0.1681425571, -0.0299745724, 0.0486975536, -0.0063061565, 0.0236783326, 0.0341806151, -0.0060020285, 0.0389456861, -0.0397201478, -0.0235554166, -0.0422564894, 0.0034768924, 0.0088781826, -0.0178075843, 0.0194100142, 0.0192374270, 0.0500968285, 0.0213430095, 0.0544736125, -0.0182459187, 0.0687645525, -0.0637109429, 0.0296506695, -0.0059191003, 0.0076175597, 0.0333553739, -0.0284797326, -0.0172422417, 0.0304772016, -0.0203301460, 0.0091125453, 0.0124873500, -0.0118052047, 0.0196281988, -0.1005409285, -0.0486363582, -0.0571581423, -0.0057667587, -0.0229607932, -0.0243438911, -0.0022394953, -0.0382377729, 0.0186608210, -0.0510486737, -0.0131222252, -0.0002630295, -0.0244874880, 0.0080776270, 0.0183746256, -0.0133937960, 0.0016614854, 0.0248861276, -0.0089258626, -0.0107191987, 0.1273662001, 0.0048037404, 0.0793801546, 0.0583712608, -0.0793141648, -0.0006869423, 0.0132788960, -0.0050497456, -0.0006381508, -0.0005771907, 0.0000142747, -0.0003775199};

    ModelOutput *net_outputs = reinterpret_cast<ModelOutput *>(input_array);
    if (!net_outputs) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "Invalid net_outputs pointer");
        return -1;
    }

    ModelMap data;
    ConfidenceClass confidence = GREEN;
    fill_model(data, confidence, *net_outputs);

    // Update preallocated jfloatArray objects
    const char *keys[] = {
            "position_t", "position_x", "position_y", "position_z", "position_x_std",
            "position_y_std", "position_z_std",
            "velocity_t", "velocity_x", "velocity_y", "velocity_z",
            "acceleration_t", "acceleration_x", "acceleration_y", "acceleration_z",
            "orientation_t", "orientation_x", "orientation_y", "orientation_z",
            "orientation_rate_t", "orientation_rate_x", "orientation_rate_y", "orientation_rate_z",
            "lane_lines_t", "lane_lines_x",
            "lane_lines_y0", "lane_lines_y1", "lane_lines_y2", "lane_lines_y3",
            "lane_lines_z0", "lane_lines_z1", "lane_lines_z2", "lane_lines_z3",
            "lane_line_stds", "lane_line_probs",
            "road_edges_t", "road_edges_x", "road_edges_y0", "road_edges_y1",
            "road_edges_z0", "road_edges_z1", "road_edge_stds",
            "leads_t", "leads_prob", "leads_prob_time",
            "leads_x0", "leads_y0", "leads_v0", "leads_a0",
            "leads_x_std0", "leads_y_std0", "leads_v_std0", "leads_a_std0",
            "leads_x1", "leads_y1", "leads_v1", "leads_a1",
            "leads_x_std1", "leads_y_std1", "leads_v_std1", "leads_a_std1",
            "leads_x2", "leads_y2", "leads_v2", "leads_a2",
            "leads_x_std2", "leads_y_std2", "leads_v_std2", "leads_a_std2",
            "disengage_t", "gas_disengage_probs", "brake_disengage_probs", "steer_override_probs",
            "brake_3ms2_probs", "brake_4ms2_probs", "brake_5ms2_probs",
            "engaged_prob", "desire_prediction", "desire_state", "hard_brake_predicted",
            "temporal_trans", "temporal_rot", "temporal_trans_std", "temporal_rot_std"
    };

    const jsize expectedSizes[] = {
            33, 33, 33, 33, 33, 33, 33, // position_t to position_z_std
            33, 33, 33, 33, // velocity_t to velocity_z
            33, 33, 33, 33, // acceleration_t to acceleration_z
            33, 33, 33, 33, // orientation_t to orientation_z
            33, 33, 33, 33, // orientation_rate_t to orientation_rate_z
            33, 33, // lane_lines_t, lane_lines_x
            33, 33, 33, 33, // lane_lines_y0 to lane_lines_y3
            33, 33, 33, 33, // lane_lines_z0 to lane_lines_z3
            4, 4, // lane_line_stds, lane_line_probs
            33, 33, 33, 33, // road_edges_t to road_edges_y1
            33, 33, 2, // road_edges_z0, road_edges_z1, road_edge_stds
            6, 3, 3, // leads_t, leads_prob, leads_prob_time
            6, 6, 6, 6, // leads_x0, leads_y0, leads_v0, leads_a0
            6, 6, 6, 6, // leads_x_std0, leads_y_std0, leads_v_std0, leads_a_std0
            6, 6, 6, 6, // leads_x1, leads_y1, leads_v1, leads_a1
            6, 6, 6, 6, // leads_x_std1, leads_y_std1, leads_v_std1, leads_a_std1
            6, 6, 6, 6, // leads_x2, leads_y2, leads_v2, leads_a2
            6, 6, 6, 6, // leads_x_std2, leads_y_std2, leads_v_std2, leads_a_std2
            5, 5, 5, 5, // disengage_t to steer_override_probs
            5, 5, 5, // brake_3ms2_probs to brake_5ms2_probs
            1, 32, 8, 1, // engaged_prob, desire_prediction, desire_state, hard_brake_predicted
            3, 3, 3, 3 // temporal_trans to temporal_rot_std
    };
    jsize poolSize = env->GetArrayLength(arrayPool);
    if (poolSize != 84) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                      "Array pool size mismatch");
        return -1;
    }

    for (jsize i = 0; i < poolSize; ++i) {
        jfloatArray array = (jfloatArray) env->GetObjectArrayElement(arrayPool, i);
        jsize len = env->GetArrayLength(array);
        if (len != expectedSizes[i]) {
            __android_log_print(ANDROID_LOG_ERROR, "CameraManager",
                                " 22 Array size mismatch for key %s at index %d: expected %d, got %d",
                                keys[i], i, expectedSizes[i], len);

            env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"),
                          "Array size o");
            env->DeleteLocalRef(array);
            return -1;
        }
        auto it = data.find(keys[i]);
        if (it != data.end()) {
            env->SetFloatArrayRegion(array, 0, it->second.size(), it->second.data());
        }
        env->DeleteLocalRef(array);
    }


    return static_cast<jint>(confidence);


    return 8;
}
}


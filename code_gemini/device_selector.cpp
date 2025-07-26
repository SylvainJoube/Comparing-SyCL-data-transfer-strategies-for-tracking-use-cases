#include <sycl/sycl.hpp>
#include <iostream>
#include <string>
#include <vector>

// --- Mock Global State & Helpers (replace with your actual code) ---
// This struct and variables represent the global state your old code was modifying.
struct RuntimeEnvironment {
    std::string computer_name;
    std::string device_name;
    int device_score = 0;
};
RuntimeEnvironment runtime_environment;
std::string MUST_RUN_ON_DEVICE_NAME;

void log(const std::string& message) {
    std::cout << "[LOG] " << message << std::endl;
}
// ---

/**
 * @brief Generates a stable, unique score for a given SYCL device.
 *
 * This function creates a score based on the platform and device index,
 * ensuring it's the same every time the program runs.
 *
 * @param device The SYCL device.
 * @return A stable integer score.
 */
int get_stable_device_score(const sycl::device& device) {
    auto platform = device.get_platform();
    auto all_platforms = sycl::platform::get_platforms();
    auto all_devices = platform.get_devices();

    // Find the index of the platform and device
    int platform_idx = 0;
    for (const auto& p : all_platforms) {
        if (p == platform) break;
        platform_idx++;
    }

    int device_idx = 0;
    for (const auto& d : all_devices) {
        if (d == device) break;
        device_idx++;
    }

    // Create a unique score, e.g., platform * 100 + device
    return (platform_idx + 1) * 100 + device_idx;
}

/**
 * @brief Lists all available SYCL devices and their stable scores.
 *
 * This function iterates through all platforms and devices, printing their
 * details to the console so the user can make a selection.
 */
void list_available_devices() {
    log("Discovering available devices...");
    auto platforms = sycl::platform::get_platforms();
    if (platforms.empty()) {
        log("No SYCL platforms found.");
        return;
    }

    for (const auto& platform : platforms) {
        log("Platform: " + platform.get_info<sycl::info::platform::name>());
        auto devices = platform.get_devices();
        if (devices.empty()) {
            log("  No devices found on this platform.");
            continue;
        }
        for (const auto& device : devices) {
            std::string dev_type_str = "Unknown";
            auto dev_type = device.get_info<sycl::info::device::device_type>();
            if (dev_type == sycl::info::device_type::gpu) dev_type_str = "GPU";
            if (dev_type == sycl::info::device_type::cpu) dev_type_str = "CPU";
            if (dev_type == sycl::info::device_type::accelerator) dev_type_str = "Accelerator";

            std::cout << "  - Device: " << device.get_info<sycl::info::device::name>()
                      << " (" << dev_type_str << ")"
                      << " -> Score: " << get_stable_device_score(device)
                      << std::endl;
        }
    }
}

/**
 * @brief A SYCL 2020-compliant device selector that chooses a device
 * based on a stable score.
 */
struct ScoredDeviceSelector {
private:
    int target_score;

public:
    // Constructor takes the score of the desired device.
    ScoredDeviceSelector(int score) : target_score(score) {}

    // The operator() is called by the SYCL runtime for each available device.
    int operator()(const sycl::device& device) const {
        int current_device_score = get_stable_device_score(device);

        // If the device's score matches our target, select it.
        if (current_device_score == target_score) {
            std::string devName = device.get_info<sycl::info::device::name>();
            log("Device selected: " + devName);

            // --- Update global state (as the original code did) ---
            runtime_environment.device_name = devName;
            runtime_environment.device_score = target_score;
            MUST_RUN_ON_DEVICE_NAME = devName;
            // ---

            return 100; // Return a high score to confirm selection.
        }

        return -1; // Return -1 to reject any other device.
    }
};


int main() {
    // 1. First, list all devices so the user can see the options and scores.
    list_available_devices();

    // 2. Prompt the user to enter their choice.
    int chosen_score = 0;
    std::cout << "\n> Please enter the score of the device you want to use: ";
    std::cin >> chosen_score;

    if (std::cin.fail() || chosen_score <= 0) {
        std::cerr << "Invalid input. Please enter a valid score." << std::endl;
        return 1;
    }

    try {
        // 3. Create a selector with the user's choice and create the queue.
        ScoredDeviceSelector selector(chosen_score);
        sycl::queue my_queue(selector);

        log("Successfully created queue on: " + runtime_environment.device_name);
        log("Selected device score was: " + std::to_string(runtime_environment.device_score));

    } catch (const sycl::exception& e) {
        std::cerr << "\nError: Could not create queue. A device with score '"
                  << chosen_score << "' might not exist." << std::endl;
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

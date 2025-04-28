#include <libyuv.h>

#include <iostream>
#include <optional>

#include "include/virtual_output.hpp"

class Camera {
   private:
    VirtualOutput virtual_output;

   public:
    Camera(uint32_t width, uint32_t height, [[maybe_unused]] double fps,
           uint32_t fourcc, std::optional<std::string> device_)
        : virtual_output{width, height, fourcc, device_} {}

    void close() { virtual_output.stop(); }

    std::string device() { return virtual_output.device(); }

    uint32_t native_fourcc() { return virtual_output.native_fourcc(); }

    void send(uint8_t* frame) {
        virtual_output.send(frame);
    }
};
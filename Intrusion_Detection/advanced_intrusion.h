#include "core/app.h"
#include "core/common_modules.h"

class intrusion_logic_module: public awi_module{

  public:
    ~intrusion_logic_module(){
    };

    void run(awi::meta &meta);
    awi::blob &tangent_blob(awi::track &terminated_track, awi::awiros_line &crossed_line);

};

class awiros_intrusion: public awiros_app{
      std::string detection_file = "/home/awiros-docker/alexandria/common.yolov3/onnx/v3/spec.json";
      //std::string detection_file = "/home/awiros-docker/alexandria/ppe.tiny_yolov3/onnx/v2/spec.json";
      std::string alert_frame_policy = "last_blob";

    public:
      ~awiros_intrusion(){
      };
      virtual void init();
      virtual void create_primary_model(std::string& detection_file);
      virtual void object_filter();
      virtual void init_tracker(std::string alert_frame_policy);
      virtual void visualizer();
      virtual void vfences_checker();
      virtual void intrusion_logic();
      virtual void init_annotation();
      virtual void send_event();
      virtual void play_pipeline();
      virtual void init_logger();
      virtual void run();
      virtual void init_conf_filter();
};
~  
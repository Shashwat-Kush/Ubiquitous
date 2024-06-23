#include "intrusion.h"
#include "core/model_config.h"
#include "core/model_factory.h"
#include <climits>
#include <iostream>

// writing the logic for sending the alert
void intrusion_logic_module::run(awi::meta &meta) {
    for(auto& sit : meta.streams){
      int thresh = static_cast<int>(std::ceil(10*sit.config.detection_config.fuzzyness));

      for(auto &blob : sit.tracker.curr_blobs){
        auto &track = sit.tracker.blob_hist.at(blob.get_id());
        awi::motion event_motion;
        if(track.influx_count == thresh || track.outflux_count == thresh){
          if(track.influx_count == thresh){
            track.influx_count += 1;
          }
          if(track.outflux_count == thresh){
            track.outflux_count += 1;
          }
          auto &last_blob = track.best_blob;
          auto &crossed_line = track.crossed_lines.back();
          awi::event event;
          cv::Rect coord = last_blob.get_cv_rect();
          auto frame = last_blob.bframe.frame.clone();
          int blob_id = 0;
          last_blob.set_id(blob_id);
          last_blob.myclass.id = (long)blob_id;
          event.eve_blob.push_back(last_blob);

          cv::line(frame, cv::Point(crossed_line.startp.x, crossed_line.startp.y), cv::Point(crossed_line.endp.x, crossed_line.endp.y), cv::Scalar(0, 0, 255), 4);
          cv::rectangle(frame, coord, cv::Scalar(0, 255, 0), 2);
          event.eve_frame =frame;
          event_motion.lines.push_back(crossed_line);
          event.eve_motion = event_motion;
          sit.events.push_back(event);

          // Track front cx,cy will be curr_blob cx, cy
          awi::blob &b_start = track.hist[0];
          track.influx_count = 0;
          track.outflux_count = 0;
          b_start.cx = blob.cx;
          b_start.cy = blob.cy;

        }

        if(track.region_count == thresh ){
          auto &last_blob = track.best_blob;
          track.region_count += 1;
          auto &crossed_region = track.crossed_regions.back();
          awi::event event;
          cv::Rect coord = last_blob.get_cv_rect();
          auto frame = last_blob.bframe.frame.clone();
          int blob_id = 0;
          last_blob.set_id(blob_id);
          last_blob.myclass.id = (long)blob_id;
          event.eve_blob.push_back(last_blob);
          cv::Rect region_coord = crossed_region.get_bounding_rect();
          cv::rectangle(frame, region_coord, cv::Scalar(0, 0, 255), 2);

          //cv::line(frame, cv::Point(crossed_line.startp.x, crossed_line.startp.y), cv::Point(crossed_line.endp.x, crossed_line.endp.y), cv::Scalar(0, 0, 255), 4);
          cv::rectangle(frame, coord, cv::Scalar(0, 255, 0), 2);
          event_motion.regions.push_back(crossed_region);
          event.eve_frame =frame;
          event.eve_motion = event_motion;
          sit.events.push_back(event);
        }
      }
      //delete tracks
   }
}

void awiros_intrusion::init(){

  static awi::base_app_status_module b_status = awi::base_app_status_module();
  b_status.name = "bstatus";
  b_status.register_(this);

  static awi::base_camera_module b_cam(this->meta);
  b_cam.name = "bcam";
  b_cam.register_(this);

  awi::base_camera_fps_module* b_fps = new awi::base_camera_fps_module(10, false, this);
  b_fps->name = "bfps";
  b_fps->register_(this);

}

void awiros_intrusion::create_primary_model(std::string &spec_path){
  const std::string &provider = this->meta.app_type;
  awi::debug("Setting confidence for model config");
  const float confidence = this->meta.p_config.config.confidence;
  awi::debug("Setting Confidence Value at: ", confidence);

  awi::ModelConfig config(spec_path, provider, this->meta.streams.size(), confidence);
  static awi_module *object_detector = awi::ModelFactory("onnx").get_model("Yolov3", config);
  object_detector->name = "bobjectdetector";
  object_detector->register_(this);

}

void awiros_intrusion::init_conf_filter(){
  static awi::confidence_filter_module module = awi::confidence_filter_module();
  module.name = "bconfidencefilter";
  module.register_(this);
}

void awiros_intrusion::object_filter(){

  awi::object_filtering_module* person_filter = new awi::object_filtering_module("person");
  person_filter->name = "filter";
  person_filter->register_(this);
}

void awiros_intrusion::vfences_checker() {
  auto *b_lines = new awi::line_crossing_module(this->meta, true);
  // auto *b_lines = new awi::line_crossing_module(this->meta, false);
   auto *b_regions = new awi::region_crossing_module(this->meta, true);
  // auto *b_regions = new awi::region_crossing_module(this->meta, false);

  b_lines->name = "blines";
  b_regions->name = "bregions";
  b_lines->register_(this);
  b_regions->register_(this);
}

void awiros_intrusion::intrusion_logic(){
  intrusion_logic_module* b_vfence_logic = new intrusion_logic_module();
  b_vfence_logic->name = "blogic";
  b_vfence_logic->register_(this);
}

void awiros_intrusion::init_tracker(std::string alert_frame_policy){

  static awi::tracking_module b_tracker(this->meta, "kalman");
  b_tracker.alert_frame_policy = alert_frame_policy;
  b_tracker.name = "btracker";
  b_tracker.register_(this);
}

void awiros_intrusion::init_annotation(){

  awi::live_annotation_module* b_annotation = new awi::live_annotation_module(this->meta, "Intrusion", true);
  b_annotation->name = "bannotation";
  b_annotation->register_(this);
}

void awiros_intrusion::visualizer(){

  awi::tracking_visualizer_module *b_visualizer = new awi::tracking_visualizer_module(this->meta);
  b_visualizer->name="bvisualizer";
  b_visualizer->register_(this);
}
void awiros_intrusion::send_event(){

    awi::send_event_module* send_events = new awi::send_event_module();
    send_events->name = "bsendevents";
    send_events->register_(this);
}

void awiros_intrusion::play_pipeline(){

    while(1){

      if (this->meta.streams.size()==0)
        return;

      for(auto& module : this->common_modules){
        module->run(this->meta);
      }
    }
}

void awiros_intrusion::init_logger(){
        static awi::log_to_file_module b_logger(this->meta, 10);
        b_logger.register_(this);
}








void awiros_intrusion::run(){
  this->init();
  this->create_primary_model(detection_file);

  this->object_filter(); // filtering
  this->init_tracker(this->alert_frame_policy);
  this->vfences_checker();
  this->intrusion_logic();
  //this->init_logger();
  this->send_event();
  this->play_pipeline();
}
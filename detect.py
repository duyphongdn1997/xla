from imageai.Detection import VideoObjectDetection
import os
import cv2
execution_path = os.getcwd()

#camera = cv2.VideoCapture(0)
"""def forFrame(frame_number, output_array, output_count):
    log_frame = open("log_frame.txt","a+")
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")
    log_frame.write(str(frame_number))
    log_frame.write(str(output_array))
    log_frame.write(str(output_count))
    log_frame.write("\n------------END OF A FRAME --------------\n")
    log_frame.close()
"""
def forSeconds(second_number, output_arrays, count_arrays, average_output_count, detected_frame):
    log_Sec = open("log_sec.txt","a+")
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print(count_arrays[1]['car'])
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("Returned Objects is : ", type(detected_frame))
    print("------------END OF A SECOND --------------")
    log_Sec.write(str(second_number))
    log_Sec.write(str(output_arrays))
    log_Sec.write(str(count_arrays))
    log_Sec.write(str(average_output_count))
    #log_Sec.write(str(this_second_counting))
    #log_Sec.write("\n------------END OF A SECOND --------------\n")
    log_Sec.close()
def forMinute(minute_number, output_arrays, count_arrays, average_output_count,):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed="fast")

#print(video_path)
#custom_objects = detector.CustomObjects(person=True, bicycle=True, motorcycle=True, car=True )
#
#video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, input_file_path=os.path.join(execution_path, "traffic.mp4"),
#                                output_file_path=os.path.join(execution_path, "traffic_custom_detected")
#                                , frames_per_second=20, log_progress=True,frame_detection_interval=5)

#video_path = detector.detectObjectsFromVideo(camera_input=camera,frame_detection_interval=5,
#                                output_file_path=os.path.join(execution_path, "camera_detected_video"),
#                                frames_per_second=10, per_second_function = forSeconds, per_minute_function= forMinute, minimum_percentage_probability=30)
video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"),
                                output_file_path=os.path.join(execution_path, "camera_detected_video"),
                                frames_per_second=10, per_second_function = forSeconds, per_minute_function= forMinute, minimum_percentage_probability=50,
                                             return_detected_frame = True)

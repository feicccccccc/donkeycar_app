#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent)] [--camera=(single|stereo)] [--meta=<key:value> ...]
    manage.py (train) [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)] [--continuous] [--aug]


Options:
    -h --help          Show this screen.
    --js               Use physical joystick.
    -f --file=<file>   A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value> Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
"""

# LOG: LSTM_imu model hardcoded part: bin parameters in input and output of keras part
import os
import time

from docopt import docopt
import numpy as np

import donkeycar as dk

# import parts
from donkeycar.parts.transform import Lambda, TriggeredCallback, DelayedTrigger
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, JoystickController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.utils import *


def drive(cfg, model_path=None, use_joystick=False, model_type=None, camera_type='single', meta=[]):

    """
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    """

    if model_type is None:
        if cfg.TRAIN_LOCALIZER:
            model_type = "localizer"
        elif cfg.TRAIN_BEHAVIORS:
            model_type = "behavior"
        else:
            model_type = cfg.DEFAULT_MODEL_TYPE

    # Initialize car
    v = dk.vehicle.Vehicle()

    # Add Camera parts
    # Input source: None
    # Output source: ctr, ImgPreProcess, Tub, ImgArrToJpg

    print("cfg.CAMERA_TYPE", cfg.CAMERA_TYPE)
    inputs = []
    threaded = True

    from donkeycar.parts.camera import CSICamera
    cam = CSICamera(image_w=cfg.IMAGE_W,
                    image_h=cfg.IMAGE_H,
                    image_d=cfg.IMAGE_DEPTH,
                    framerate=cfg.CAMERA_FRAMERATE,
                    gstreamer_flip=cfg.CSIC_CAM_GSTREAMER_FLIP_PARM)

    v.add(cam, inputs=inputs, outputs=['cam/image_array'], threaded=threaded)

    # Add Controller Parts
    # Input source: cam
    # Output source: DriveMode, Tub, th_filter, PilotCondition, AIRecordingCondition

    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        # modify max_throttle closer to 1.0 to have more power
        # modify steering_scale lower than 1.0 to have less responsive steering
        from donkeycar.parts.controller import get_js_controller

        ctr = get_js_controller(cfg)

        if cfg.USE_NETWORKED_JS:
            from donkeycar.parts.controller import JoyStickSub
            netwkJs = JoyStickSub(cfg.NETWORK_JS_SERVER_IP)
            v.add(netwkJs, threaded=True)
            ctr.js = netwkJs

    else:
        # This web controller will create a web server that is capable
        # of managing steering, throttle, and modes, and more.
        ctr = LocalWebController()

    v.add(ctr,
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    # Add one tap reverse part
    th_filter = ThrottleFilter()
    v.add(th_filter, inputs=['user/throttle'], outputs=['user/throttle'])

    # See if we should even run the pilot module.
    # This is only needed because the part run_condition only accepts boolean
    class PilotCondition:
        def run(self, mode):
            if mode == 'user':
                return False
            else:
                return True

    v.add(PilotCondition(), inputs=['user/mode'], outputs=['run_pilot'])

    def get_record_alert_color(num_records):
        col = (0, 0, 0)
        for count, color in cfg.RECORD_ALERT_COLOR_ARR:
            if num_records >= count:
                col = color
        return col

    class RecordTracker:
        def __init__(self):
            self.last_num_rec_print = 0
            self.dur_alert = 0
            self.force_alert = 0

        def run(self, num_records):
            if num_records is None:
                return 0

            if self.last_num_rec_print != num_records or self.force_alert:
                self.last_num_rec_print = num_records

                if num_records % 10 == 0:
                    print("recorded", num_records, "records")

                if num_records % cfg.REC_COUNT_ALERT == 0 or self.force_alert:
                    self.dur_alert = num_records // cfg.REC_COUNT_ALERT * cfg.REC_COUNT_ALERT_CYC
                    self.force_alert = 0

            if self.dur_alert > 0:
                self.dur_alert -= 1

            if self.dur_alert != 0:
                return get_record_alert_color(num_records)

            return 0

    rec_tracker_part = RecordTracker()
    v.add(rec_tracker_part, inputs=["tub/num_records"], outputs=['records/alert'])

    if cfg.AUTO_RECORD_ON_THROTTLE and isinstance(ctr, JoystickController):
        # then we are not using the circle button. hijack that to force a record count indication
        def show_record_account_status():
            rec_tracker_part.last_num_rec_print = 0
            rec_tracker_part.force_alert = 1

        ctr.set_button_down_trigger('circle', show_record_account_status)

    # Add IMU parts
    from donkeycar.parts.imu import Mpu6050
    imu1 = Mpu6050(0x68)
    v.add(imu1, outputs=['imu1/acl_x', 'imu1/acl_y', 'imu1/acl_z',
                         'imu1/gyr_x', 'imu1/gyr_y', 'imu1/gyr_z'], threaded=True)
    imu2 = Mpu6050(0x69)
    v.add(imu2, outputs=['imu2/acl_x', 'imu2/acl_y', 'imu2/acl_z',
                         'imu2/gyr_x', 'imu2/gyr_y', 'imu2/gyr_z'], threaded=True)

    class ImgPreProcess():
        """
        preprocess camera image for inference.
        normalize and crop if needed.
        """

        def __init__(self, cfg):
            self.cfg = cfg

        def run(self, img_arr):
            return normalize_and_crop(img_arr, self.cfg)

    # inference input, Normalised and cropped with ROI
    inf_input = 'cam/normalized/cropped'
    v.add(ImgPreProcess(cfg),
          inputs=['cam/image_array'],
          outputs=[inf_input],
          run_condition='run_pilot')


    class TimeSequenceFrames_img:
        '''
        Input to LSTM
        Return frame dimension (1,7,120,160,4)
        '''

        def __init__(self, num_states=7):
            self.rnn_input = None
            self.num_states = num_states  # Number of States for RNN

        def run(self, img):

            if img is None:
                return img

            if self.rnn_input is None:
                self.rnn_input = np.stack(([img] * self.num_states), axis=0)
            else:
                img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
                self.rnn_input = np.append(self.rnn_input[1:self.num_states,:, :, :],img , axis=0)
            #print("img_seq: {}".format(self.rnn_input))

            return self.rnn_input

    class TimeSequenceFrames_imu:
        '''
        Input to LSTM
        Return frame dimension (1,7,12)
        '''

        def __init__(self, num_states=7):
            self.rnn_input = None
            self.num_states = num_states  # Number of States for RNN

        def run(self,
                accel_x1, accel_y1, accel_z1, gyr_x1, gyr_y1, gyr_z1,
                accel_x2, accel_y2, accel_z2, gyr_x2, gyr_y2, gyr_z2):

            imu_arr = np.array([accel_x1, accel_y1, accel_z1, gyr_x1, gyr_y1, gyr_z1,
                       accel_x2, accel_y2, accel_z2, gyr_x2, gyr_y2, gyr_z2])

            if self.rnn_input is None:
                self.rnn_input = np.stack(([imu_arr] * self.num_states), axis=0)
            else:
                imu_arr = imu_arr.reshape(1, imu_arr.shape[0])
                self.rnn_input = np.append(self.rnn_input[1:self.num_states],imu_arr, axis=0)
            
            #print("imu_seq: {}".format(self.rnn_input)) 
            return self.rnn_input

    class TimeSequenceFrames_prev_input:
        '''
        Input to LSTM
        Return frame dimension (1,5,2)
        '''

        def __init__(self, num_states=5):
            self.rnn_input = None
            self.num_states = num_states  # Number of States for RNN

        def run(self,
                accel_x1, accel_y1, accel_z1, gyr_x1, gyr_y1, gyr_z1,
                accel_x2, accel_y2, accel_z2, gyr_x2, gyr_y2, gyr_z2):

            imu_arr = np.array([accel_x1, accel_y1, accel_z1, gyr_x1, gyr_y1, gyr_z1,
                                accel_x2, accel_y2, accel_z2, gyr_x2, gyr_y2, gyr_z2])

            if self.rnn_input is None:
                self.rnn_input = np.stack(([imu_arr] * self.num_states), axis=0)
            else:
                imu_arr = imu_arr.reshape(1, imu_arr.shape[0])
                self.rnn_input = np.append(self.rnn_input[1:self.num_states], imu_arr, axis=0)

            # print("imu_seq: {}".format(self.rnn_input))
            return self.rnn_input

    # model input

    inputs = [inf_input,
              'imu1/acl_x', 'imu1/acl_y', 'imu1/acl_z',
              'imu1/gyr_x', 'imu1/gyr_y', 'imu1/gyr_z',
              'imu2/acl_x', 'imu2/acl_y', 'imu2/acl_z',
              'imu2/gyr_x', 'imu2/gyr_y', 'imu2/gyr_z']

    if model_type == "rnn_imu" or \
            model_type == 'rnn_imu_linear' or \
            model_type == 'rnn_imu_many2many' or \
            model_type == 'rnn_imu_many2many_imupred' or \
            model_type == "test":

        img_ts_frames = TimeSequenceFrames_img(num_states=cfg.SEQUENCE_LENGTH)
        v.add(img_ts_frames, inputs=['cam/normalized/cropped'], outputs=['cam/ts_frames'])
        imu_ts_frames = TimeSequenceFrames_imu(num_states=cfg.SEQUENCE_LENGTH)
        v.add(imu_ts_frames,
              inputs=['imu1/acl_x', 'imu1/acl_y', 'imu1/acl_z',
                      'imu1/gyr_x', 'imu1/gyr_y', 'imu1/gyr_z',
                      'imu2/acl_x', 'imu2/acl_y', 'imu2/acl_z',
                      'imu2/gyr_x', 'imu2/gyr_y', 'imu2/gyr_z'],
              outputs=['imu/ts_frames'])

        inputs = ['cam/ts_frames', 'imu/ts_frames']

    if model_type == "test":
        control_ts_frame = TimeSequenceFrames_prev_input(num_states=cfg.SEQUENCE_LENGTH)
        v.add(control_ts_frame, inputs=['pilot/angle', 'pilot/throttle'], outputs=['pilot/angle_frames', 'pilot/throttle_frames'])
        inputs = ['cam/ts_frames', 'imu/ts_frames', 'pilot/angle_frames', 'pilot/throttle_frames']


    def load_model(kl, model_path):
        start = time.time()
        print('loading model', model_path)
        kl.load(model_path)
        print('finished loading in %s sec.' % (str(time.time() - start)))

    def load_weights(kl, weights_path):
        start = time.time()
        try:
            print('loading model weights', weights_path)
            kl.model.load_weights(weights_path)
            print('finished loading in %s sec.' % (str(time.time() - start)))
        except Exception as e:
            print(e)
            print('ERR>> problems loading weights', weights_path)

    def load_model_json(kl, json_fnm):
        start = time.time()
        print('loading model json', json_fnm)
        from tensorflow.python import keras
        try:
            with open(json_fnm, 'r') as handle:
                contents = handle.read()
                kl.model = keras.models.model_from_json(contents)
            print('finished loading json in %s sec.' % (str(time.time() - start)))
        except Exception as e:
            print(e)
            print("ERR>> problems loading model json", json_fnm)

    if model_path:
        # When we have a model, first create an appropriate Keras part
        kl = dk.utils.get_model_by_type(model_type, cfg)

        model_reload_cb = None

        if '.h5' in model_path or '.uff' in model_path or 'tflite' in model_path or '.pkl' in model_path:
            # when we have a .h5 extension
            # load everything from the model file
            load_model(kl, model_path)

            def reload_model(filename):
                load_model(kl, filename)

            model_reload_cb = reload_model

        elif '.json' in model_path:
            # when we have a .json extension
            # load the model from there and look for a matching
            # .wts file with just weights
            load_model_json(kl, model_path)
            weights_path = model_path.replace('.json', '.weights')
            load_weights(kl, weights_path)

            def reload_weights(filename):
                weights_path = filename.replace('.json', '.weights')
                load_weights(kl, weights_path)

            model_reload_cb = reload_weights

        else:
            print("ERR>> Unknown extension type on model file!!")
            return

        # this part will signal visual LED, if connected
        v.add(FileWatcher(model_path, verbose=True), outputs=['modelfile/modified'])

        # these parts will reload the model file, but only when ai is running so we don't interrupt user driving
        v.add(FileWatcher(model_path), outputs=['modelfile/dirty'], run_condition="ai_running")
        v.add(DelayedTrigger(100), inputs=['modelfile/dirty'], outputs=['modelfile/reload'], run_condition="ai_running")
        v.add(TriggeredCallback(model_path, model_reload_cb), inputs=["modelfile/reload"], run_condition="ai_running")

        outputs = ['pilot/angle', 'pilot/throttle']

        if cfg.TRAIN_LOCALIZER:
            outputs.append("pilot/loc")

        v.add(kl, inputs=inputs,
              outputs=outputs,
              run_condition='run_pilot')

        # Choose what inputs should change the car.

    # Output the actual output base on user/mode
    # Add the Mode selection parts

    class DriveMode:
        def run(self, mode,
                user_angle, user_throttle,
                pilot_angle, pilot_throttle):
            if mode == 'user':
                return user_angle, user_throttle

            elif mode == 'local_angle':
                return pilot_angle, user_throttle

            else:
                # 'local_pilot'
                return pilot_angle, pilot_throttle * cfg.AI_THROTTLE_MULT

    v.add(DriveMode(),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'throttle'])


    # # to give the car a boost when starting ai mode in a race.
    # # Not quite useful if we are doing drift here
    # ai_launcher = AiLaunch(cfg.AI_LAUNCH_DURATION, cfg.AI_LAUNCH_THROTTLE, cfg.AI_LAUNCH_KEEP_ENABLED)
    #
    # v.add(ai_launcher,
    #       inputs=['user/mode', 'throttle'],
    #       outputs=['throttle'])
    #
    # if isinstance(ctr, JoystickController):
    #     ctr.set_button_down_trigger(cfg.AI_LAUNCH_ENABLE_BUTTON, ai_launcher.enable_ai_launch)
    #
    # class AiRunCondition:
    #     '''
    #     A bool part to let us know when ai is running.
    #     '''
    #
    #     def run(self, mode):
    #         if mode == "user":
    #             return False
    #         return True
    #
    # v.add(AiRunCondition(), inputs=['user/mode'], outputs=['ai_running'])

    # Ai Recording / Record while driving on the AI, probably for RL / self-supervised learning
    # Input source: ctr
    # output: Tub_runCondition

    # class AiRecordingCondition:
    #     """
    #     return True when ai mode, otherwise respect user mode recording flag
    #     """
    #
    #     def run(self, mode, recording):
    #         if mode == 'user':
    #             return recording
    #         return True
    #
    # if cfg.RECORD_DURING_AI:
    #     v.add(AiRecordingCondition(), inputs=['user/mode', 'recording'], outputs=['recording'])

    # Add the PCA9685 to control physical parts

    from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

    steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    steering = PWMSteering(controller=steering_controller,
                           left_pulse=cfg.STEERING_LEFT_PWM,
                           right_pulse=cfg.STEERING_RIGHT_PWM)

    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    throttle = PWMThrottle(controller=throttle_controller,
                           max_pulse=cfg.THROTTLE_FORWARD_PWM,
                           zero_pulse=cfg.THROTTLE_STOPPED_PWM,
                           min_pulse=cfg.THROTTLE_REVERSE_PWM)

    v.add(steering, inputs=['angle'])
    v.add(throttle, inputs=['throttle'])

    # Add Tub to save data

    inputs = ['cam/image_array',
              'user/angle', 'user/throttle',
              'user/mode',
              'imu1/acl_x', 'imu1/acl_y', 'imu1/acl_z',
              'imu1/gyr_x', 'imu1/gyr_y', 'imu1/gyr_z',
              'imu2/acl_x', 'imu2/acl_y', 'imu2/acl_z',
              'imu2/gyr_x', 'imu2/gyr_y', 'imu2/gyr_z'
              ]

    types = ['image_array',
             'float', 'float',
             'str',
             'float', 'float', 'float',
             'float', 'float', 'float',
             'float', 'float', 'float',
             'float', 'float', 'float'
             ]

    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
    v.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    # TCP services to publish camera?

    # if cfg.PUB_CAMERA_IMAGES:
    #     from donkeycar.parts.network import TCPServeValue
    #     from donkeycar.parts.image import ImgArrToJpg
    #     pub = TCPServeValue("camera")
    #     v.add(ImgArrToJpg(), inputs=['cam/image_array'], outputs=['jpg/bin'])
    #     v.add(pub, inputs=['jpg/bin'])

    if type(ctr) is LocalWebController:
        print("You can now go to <your pis hostname.local>:8887 to drive your car.")
    elif isinstance(ctr, JoystickController):
        print("You can now move your joystick to drive your car.")
        # tell the controller about the tub
        # for erasing the Tub record
        ctr.set_tub(tub)

        if cfg.BUTTON_PRESS_NEW_TUB:
            def new_tub_dir():
                v.parts.pop()
                tub = th.new_tub_writer(inputs=inputs, types=types, user_meta=meta)
                v.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')
                ctr.set_tub(tub)

            ctr.set_button_down_trigger('cross', new_tub_dir)
        ctr.print_controls()

    # run the vehicle
    v.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    if args['drive']:
        model_type = args['--type']
        camera_type = args['--camera']
        drive(cfg, model_path=args['--model'], use_joystick=args['--js'], model_type=model_type,
              camera_type=camera_type,
              meta=args['--meta'])

    if args['train']:
        from train import multi_train, preprocessFileList

        tub = args['--tub']
        model = args['--model']
        transfer = args['--transfer']
        model_type = args['--type']
        continuous = args['--continuous']
        aug = args['--aug']

        dirs = preprocessFileList(args['--file'])
        if tub is not None:
            tub_paths = [os.path.expanduser(n) for n in tub.split(',')]
            dirs.extend(tub_paths)

        multi_train(cfg, dirs, model, transfer, model_type, continuous, aug)


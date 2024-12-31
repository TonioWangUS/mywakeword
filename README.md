# mywakeword

  481  python detect_from_microphone.py 
  482  ls
  483  cd ..
  484  find . -type f|grep -i download
  485  cd examples/
  486  vim download.py
  487  python download.py 
  488  ls
  489  find . -name alexa_v0.1.tflite
  490  cd ..
  491  find . -name alexa_v0.1.tflite
  492  cd examples/
  493  python detect_from_microphone.py 
  494  ls
  495  ls /home/tsna/.pyenv/versions/3.7.17/lib/python3.7/site-packages/openwakeword/resources/models
  496  python detect_from_microphone.py --inference_framework=onnx
  497  vim mynxx.py
  498  python mynxx.py 
  499  python capture_activations.py --inference_framework onnx --output_dir .
  500  ls
  501  python capture_activations.py --inference_framework onnx --output_dir ./
  502  python capture_activations.py --inference_framework onnx --output_dir .
  503  python capture_activations.py --inference_framework onnx --output_dir ./
  504  ls /home/tsna/.pyenv/versions/3.7.17/lib/python3.7/site-packages/openwakeword/resources/models
  505  ls
  506  vim mynxx.py 
  507  pyton mynxx.py 
  508  python mynxx.py 
  509  ls
  510  mkdir dataset/wake_word -p
  511  mkdir dataset/background -p
  512  mkdir dataset/silence -p
  513  cp ~/sharedvm/wakewords/* dataset/wake_word/
  514  vim train_wav.py
  515  cp dataset/wake_word/hey_chuangda.wav wake1.wav
  516  vim train_wav.py
  517  python train_wav.py 
  518  pip install librosa
  519  python train_wav.py 
  520  pip install matplotlib
  521  python train_wav.py 
  522  ls
  523  vim train_model.py
  524  python train_model.py 
  525  pip install tensorflow
  526  python train_model.py 
  527  mkdir spectrograms
  528  vim train_openwakeword.py
  529  python train_openwakeword.py 
  530  cd dataset/
  531  ls
  532  cd wake_word/
  533  ls
  534  ffmpeg -i street_noise.mp3 -o street_noise.wav
  535  ffmpeg -i street_noise.mp3  street_noise.wav
  536  aplay street_noise.wav 
  537  ffmpeg -i nature_noise.mp3 nature_noise.wav
  538  aplay nature_noise.wav 
  539  mv nature_noise.wav ../background/
  540  mv street_noise.wav ../background/
  541  ls
  542  rm *.mp3
  543  ls
  544  mv negative_wake_up_with_street_noise.wav ../silence/
  545  cd ../..
  546  ls
  547  python train_openwakeword.py 
  548  vim train_2openwakeword.py
  549  python train_2openwakeword.py 
  550  vim train_openww_shape.py
  551  python train_openww_shape.py 
  552  vim train_openww_onnx.py
  553  pyton train_openww_onnx.py 
  554  python train_openww_onnx.py 
  555  ls
  556  find . -name *.tflite
  557  cd ..
  558  find . -name *.tflite
  559  cd examples/
  560  ls
  561  vim train_wakeword_model.py
  562  python train_wakeword_model.py 
  563  vim train_openww_convert.py
  564  python train_openww_convert.py 
  565  vim train_openww_test.py
  566  python train_openww_test.py 
  567  ls
  568  aplay wake1.wav 
  569  vim train_openww_test.py 
  570  vim train_openww_test1.py 
  571  cp wake1.wav test_audio.wav
  572  vim train_openww_test1.py 
  573  python train_openww_test1.py 
  574  vim train_openww_test2.py 
  575  python train_openww_test2.py 
  576  vim train_openww_test2.py 
  577  vim train_openww_test3.py 
  578  python train_openww_test3.py 
  579  python train_openww_test4.py 
  580  vim train_openww_test4.py 
  581  python train_openww_test4.py 
  582  vim train_openww_test4.py 
  583  ls
  584  vim detect_from_microphone.py 
  585  ls /home/tsna/.pyenv/versions/3.7.17/lib/python3.7/site-packages/openwakeword/resources/models
  586  vim train_openww_onnx.py
  587  vim train_openww_onnx1.py
  588  python train_openww_onnx1.py 
  589  pip install torch
  590  python train_openww_onnx1.py 
  591  vim train_openww_onnx1.py
  592  vim train_openww_onnx1.py +35
  593  python train_openww_onnx1.py 
  594  vim train_openww_onnx1.py +35
  595  python train_openww_onnx1.py 
  596  vim train_openww_onnx1.py +35
  597  python train_openww_onnx1.py 
  598  cp train_openww_onnx1.py train_openww_onnx2.py
  599  vim train_openww_onnx2.py 
  600  python train_openww_onnx2.py 
  601  vim train_openww_onnx2.py 
  602  python train_openww_onnx2.py 
  603  vim train_openww_onnx_export.py 
  604  python train_openww_onnx_export.py 
  605  vim train_openww_onnx_export.py 
  606  python train_openww_onnx_export.py 
  607  ls
  608  cp wakeword_model.tflite hey_chuangda.tflite
  609  cp wakeword_model.onnx hey_chuangda.onnx

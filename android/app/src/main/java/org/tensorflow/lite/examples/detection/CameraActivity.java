/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.Manifest;
import android.app.Fragment;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;
import android.support.annotation.NonNull;
import android.support.design.widget.BottomSheetBehavior;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.SwitchCompat;
import android.support.v7.widget.Toolbar;
import android.util.Size;
import android.view.Gravity;
import android.view.Surface;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import uk.me.berndporr.iirj.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.CameraConnectionFragment.interfaceCallback;

public abstract class CameraActivity extends AppCompatActivity
    implements View.OnClickListener {
  private static final Logger LOGGER = new Logger();

  private boolean debug = false;
  private Handler handler;
  private HandlerThread handlerThread;

  private LinearLayout bottomSheetLayout;
  private LinearLayout gestureLayout;
  private BottomSheetBehavior sheetBehavior;
  private ProgressDialog progressDialog;

  protected TextView frameValueTextView, cropValueTextView, inferenceTimeTextView;
  protected ImageView bottomSheetArrowImageView;
  private ImageView plusImageView, minusImageView;
  private SwitchCompat apiSwitchCompat;
  private TextView Record1,Record2,Record3;
  private TextView result_Normal,result_Crackle,result_Wheeze,result_Both;
  private TextView Normal_number,Crackle_number,Wheeze_number,Both_number;
  private Button add,delete;
  private LinearLayout mainLinerLayout;
  private boolean tag = false;
  @Override
  protected void onCreate(final Bundle savedInstanceState) {

    super.onCreate(null);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);//设置窗口始终点亮
    progressDialog = new ProgressDialog(this);
    progressDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
    progressDialog.setTitle("computing in progress");
    progressDialog.setMessage("please wait several seconds...");
    setContentView(R.layout.activity_camera);
    Toolbar toolbar = findViewById(R.id.toolbar);
    setSupportActionBar(toolbar);
    getSupportActionBar().setDisplayShowTitleEnabled(false);//TensorFlow Lite
   // setFragment();
    onPreviewSizeChosen();

    apiSwitchCompat = findViewById(R.id.api_info_switch);  //似乎也没用
    bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    gestureLayout = findViewById(R.id.gesture_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout); //底部滚动条
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);
    Record1 = findViewById(R.id.record1);
    Record1.setOnClickListener(this);
    Record2 = findViewById(R.id.record2);
    Record2.setOnClickListener(this);
    Record3 = findViewById(R.id.record3);
    Record3.setOnClickListener(this);
    result_Normal = findViewById(R.id.result_Normal);
    result_Crackle = findViewById(R.id.result_Crackle);
    result_Wheeze = findViewById(R.id.result_Wheeze);
    result_Both = findViewById(R.id.result_Both);
    Normal_number = findViewById(R.id.Normal_number);
    Wheeze_number = findViewById(R.id.Wheeze_number);
    Crackle_number = findViewById(R.id.Crackle_number);
    Both_number = findViewById(R.id.Both_number);
    mainLinerLayout = findViewById(R.id.MyTable);
    add = findViewById(R.id.add);
    add.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        read_wav();
        tag = true;
      }
    });
    delete = findViewById(R.id.delete);
    delete.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View v) {
        for(int i=5;i<11;i++){
          if(tag) {
            String sdCardDir = Environment.getExternalStorageDirectory().getPath();
            File file = new File(sdCardDir + "/LBStethoscope/cache/bin/lbstethoscope_1703/recorded/lbstethoscope_1703_1637/");
            File[] files = file.listFiles();
            for (int j = 0; j < files.length; j++) {
              File[] filesub = files[j].listFiles();
              for (int k = 0; k < filesub.length; k++) {
                filesub[k].delete();
              }
              files[j].delete();
            }
            mainLinerLayout.removeViewAt(5);
          }
        }

      }
    });




    ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
        new ViewTreeObserver.OnGlobalLayoutListener() {
          @Override
          public void onGlobalLayout() {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN) {
              gestureLayout.getViewTreeObserver().removeGlobalOnLayoutListener(this);
            } else {
              gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
            }
            //                int width = bottomSheetLayout.getMeasuredWidth();
            int height = gestureLayout.getMeasuredHeight();

            sheetBehavior.setPeekHeight(height);
          }
        });
    sheetBehavior.setHideable(false);

    sheetBehavior.setBottomSheetCallback(
        new BottomSheetBehavior.BottomSheetCallback() {//设置底部滚动条箭头方向
          @Override
          public void onStateChanged(@NonNull View bottomSheet, int newState) {
            switch (newState) {
              case BottomSheetBehavior.STATE_HIDDEN:
                break;
              case BottomSheetBehavior.STATE_EXPANDED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                }
                break;
              case BottomSheetBehavior.STATE_COLLAPSED:
                {
                  bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                }
                break;
              case BottomSheetBehavior.STATE_DRAGGING:
                break;
              case BottomSheetBehavior.STATE_SETTLING:
                bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                break;
            }
          }

          @Override
          public void onSlide(@NonNull View bottomSheet, float slideOffset) {}
        });

  }
  /***********************************分割lung sound***********************************************/
  public List<Double> find_point(List<Double> sig, int fs, int window_len) {

    double max_all = Collections.max(sig);
    for(int i=0;i<sig.size();i++){
      sig.set(i,sig.get(i)/max_all);
    }

    List<Double> output = new ArrayList<Double>();
    for(int i=0; i<sig.size(); i+=window_len) {
      int end;
      if(i+window_len<sig.size()) end = i+window_len;
      else end = sig.size()-1;
      double sum = 0;
      for(int j=i;j<end;j++) {
        sum+=sig.get(j)*sig.get(j);
      }
      output.add(sum);
    }
    double max = Collections.max(output);
    double average = 0;
    for(int i=0;i<output.size();i++) {
      output.set(i, output.get(i)/max);
      average+=output.get(i);
    }
    average = average/output.size();

    List<Integer> real_point = new ArrayList<Integer>();
    List<Integer> whole_point = new ArrayList<Integer>();
    int no_more_than_ave = 1;
    int a = 6;
    for(int i=0;i<output.size();i++) {
      if(output.get(i)>average) {
        if(no_more_than_ave==0) {
          no_more_than_ave=1;
          whole_point.add(i);
          if(real_point.isEmpty()) {
            real_point.add(i);
          }
          else {
            if(i-(int)whole_point.get(whole_point.size()-2)>a) real_point.add(i);
          }
        }
      }
      else no_more_than_ave=0;
    }
    List<Double> result = new ArrayList<Double>();
    for(int i=0;i<real_point.size();i++) {
      double mid = (double)real_point.get(i);
      double mid2 = mid*window_len/fs;
      result.add(mid2);
    }
    return result;

  }
/**************************************************************************************************/

/***********************************运行model***********************************************/
  public void run_model(List<Double> sig, List<Double> result, int fs){
    int[] number = new int[4];
    int step = 0;
    for(int i=0; i<result.size();i++){
      int len = (int)Math.floor(result.get(i)*fs)-step;

      double[] raw_data = new double[len];
      for(int j=0;j<len;j++){
        raw_data[j]=sig.get(j+step);
      }
      double[][] wavelet=wavelet(raw_data);
      double[][] STFT=stft_computer(raw_data, fs);
      int a = processImage(STFT,wavelet);
      number[a] = number[a]+1;
      /*使用raw_data运行模型*/
      step+=len;
    }
    double[] raw_data = new double[sig.size()-step];

    for(int i=step;i<sig.size();i++){
      raw_data[i-step] = sig.get(i);
    }
    double[][] wavelet=wavelet(raw_data);
    double[][] STFT=stft_computer(raw_data, fs);
    int a = processImage(STFT,wavelet);
    number[a] = number[a]+1;
    /*使用raw_data运行模型*/
    result_Normal.setText(String.valueOf(number[0])+" cycles");
    result_Crackle.setText(String.valueOf(number[1])+" cycles");
    result_Wheeze.setText(String.valueOf(number[2])+" cycles");
    result_Both.setText(String.valueOf(number[3])+" cycles");
    progressDialog.dismiss();
    LOGGER.i("output: 0:"+String.valueOf(number[0])+","+"1:"+String.valueOf(number[1])+","+"2:"+String.valueOf(number[2])+","+"3:"+String.valueOf(number[3]));
  }
/**************************************************************************************************/

  /***********************************小波变换***********************************************/
  public double[][] wavelet(double[] sig) {
    double[] cA0 = db8_cA(sig);

    cA0 = db8_cA(cA0);

    double[] cD0 = db8_cD(cA0);
    cA0 = db8_cA(cA0);

    double[] cD1 = db8_cD(cA0);

    cA0 = db8_cA(cA0);

    double[] cD2 = db8_cD(cA0);
    cA0 = db8_cA(cA0);
    double[] cD3 = db8_cD(cA0);
    cA0 = db8_cA(cA0);
    double[] cD4 = db8_cD(cA0);
    cA0 = db8_cA(cA0);
    double[] cD5 = db8_cD(cA0);
    cA0 = db8_cA(cA0);
    double[] cD6 = db8_cD(cA0);
    cA0 = db8_cA(cA0);

    double[] wavelet = new double[cD0.length+cD1.length+cD2.length+cD3.length+cD4.length+cD5.length+cD6.length+cA0.length];
    System.arraycopy(cD0,0,wavelet,0,cD0.length);
    System.arraycopy(cD1,0,wavelet,cD0.length,cD1.length);
    System.arraycopy(cD2,0,wavelet,cD0.length+cD1.length,cD2.length);
    System.arraycopy(cD3,0,wavelet,cD0.length+cD1.length+cD2.length,cD3.length);
    System.arraycopy(cD4,0,wavelet,cD0.length+cD1.length+cD2.length+cD3.length,cD4.length);
    System.arraycopy(cD5,0,wavelet,cD0.length+cD1.length+cD2.length+cD3.length+cD4.length,cD5.length);
    System.arraycopy(cD6,0,wavelet,cD0.length+cD1.length+cD2.length+cD3.length+cD4.length+cD5.length,cD6.length);
    System.arraycopy(cA0,0,wavelet,cD0.length+cD1.length+cD2.length+cD3.length+cD4.length+cD5.length+cD6.length,cA0.length);
    double max = wavelet[0];
    double min = wavelet[0];
    for (int i = 0; i < wavelet.length; i++){
      max=(wavelet[i] < max?max: wavelet[i]);
      min=(wavelet[i] > min?min: wavelet[i]);
    }

    for (int i = 0; i < wavelet.length; i++){

      wavelet[i]=(255-0)*(wavelet[i]-min)/(max-min);
    }

    int num = wavelet.length;
    int length = (int)Math.ceil(Math.sqrt(num));
    double[][] result = new double[length][length];
    for(int i=0;i<length;i++) {
      for(int j=0;j<length;j++) {
        int No = i*length+j;
        if(No<wavelet.length) result[i][j]=wavelet[No];
        else result[i][j]=0;
      }
    }

    result = resize(result);
    return result;
  }

  public double[] db8_cA(double[] sig) {

    double[] L = {-0.00011747678400228192,
            0.0006754494059985568,
            -0.0003917403729959771,
            -0.00487035299301066,
            0.008746094047015655,
            0.013981027917015516,
            -0.04408825393106472,
            -0.01736930100202211,
            0.128747426620186,
            0.00047248457399797254,
            -0.2840155429624281,
            -0.015829105256023893,
            0.5853546836548691,
            0.6756307362980128,
            0.3128715909144659,
            0.05441584224308161};

    int start = -(L.length-1);
    int end = sig.length + L.length - 1-15; //不包括end
    double[] cA = new double[end-start+1];

    for(int i=0; i<end-start+1; i=i+1) {
      cA[i] = 0;

    }
    for(int i=start;i<end; i=i+1) {
      int k=i;
      for(int j =0;j<16;j++) {
        if(k<0&&(k>=-(L.length-1))) {
          cA[i+15] = cA[i+15]+L[15-j]*sig[(-k)-1];

        }
        if(k<=sig.length-1&&k>=0) {
          cA[i+15]=cA[i+15]+L[15-j]*sig[k];

        }
        if(k<=sig.length+L.length-2&&k>sig.length-1) {
          cA[i+15]=cA[i+15]+L[15-j]*sig[2*sig.length-k-1];

        }

        k++;
      }

    }
    double[] result;
    int len;
    if (cA.length%2==1) {
      result = new double[cA.length/2];
      len = cA.length/2;
    }
    else{result = new double[cA.length/2-1];len=cA.length/2-1;}
    for(int i=0; i<len; i=i+1) {
      result[i] = cA[i*2+1];
    }
    return result;
  }
  public double[] db8_cD(double[] sig) {

    double[] H = {
            -0.05441584224308161,
            0.3128715909144659,
            -0.6756307362980128,
            0.5853546836548691,
            0.015829105256023893,
            -0.2840155429624281,
            -0.00047248457399797254,
            0.128747426620186,
            0.01736930100202211,
            -0.04408825393106472,
            -0.013981027917015516,
            0.008746094047015655,
            0.00487035299301066,
            -0.0003917403729959771,
            -0.0006754494059985568,
            -0.00011747678400228192
    };
    int start = -(H.length-1);
    int end = sig.length + H.length - 1-15; //不包括end

    double[] cD = new double[end-start+1];
    for(int i=0; i<end-start+1; i=i+1) {
      cD[i] = 0;
    }
    for(int i=start;i<end; i=i+1) {
      int k=i;
      for(int j =0;j<16;j++) {
        if(k<0&&(k>=-(H.length-1))) {
          cD[i+15] = cD[i+15]+H[15-j]*sig[(-k)-1];
        }
        if(k<=sig.length-1&&k>=0) {
          cD[i+15]=cD[i+15]+H[15-j]*sig[k];
        }
        if(k<=sig.length+H.length-2&&k>sig.length-1) {
          cD[i+15]=cD[i+15]+H[15-j]*sig[2*sig.length-k-1];
        }

        k++;
      }

    }
    double[] result;
    int len;
    if (cD.length%2==1) {
      result = new double[cD.length/2];
      len = cD.length/2;
    }
    else{result = new double[cD.length/2-1];len=cD.length/2-1;}
    for(int i=0; i<len; i=i+1) {
      result[i] = cD[i*2+1];
    }
    return result;
  }

  /********************************************************************************************/
  /***********************************STFT***********************************************/
  public static double[][] stft_computer(double[] data, int fs) {
    int n_fft = (int) (0.02 * fs);
    int win_length = n_fft;
    int hop_length = (int) (0.01 * fs);

//	    fft_window = get_window(window, win_length, fftbins=True)

//	    fft_window = util.pad_center(fft_window, n_fft)

//	    fft_window = fft_window.reshape((-1, 1))

    double[] fft_window = get_window(win_length);

//		y = np.pad(y, int(n_fft // 2), mode=pad_mode)
    double[] y = center(data, (int) Math.floor(n_fft / 2));

    // y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)
    double[][] y_frams = frame(y, n_fft, hop_length);


    double[][] stft_matrix_real = new double[(int) (1 + Math.floor(n_fft / 2))][y_frams[0].length];
    double[][] stft_matrix_imag = new double[(int) (1 + Math.floor(n_fft / 2))][y_frams[0].length];
    double[][] stft_matrix_abs = new double[(int) (1 + Math.floor(n_fft / 2))][y_frams[0].length];

    int n_columns = (int) (Math.pow(2, 8) * Math.pow(2, 10) / (8 * stft_matrix_real.length));

    for (int bl_s = 0; bl_s < stft_matrix_real[0].length; bl_s += n_columns) {

      int bl_t = Math.min(bl_s + n_columns, stft_matrix_real[0].length);

      for(int i=bl_s;i<bl_t;i++) {
        //计算stft_matrix[:,bl_s:bl_t]
        double[] mid = new double[fft_window.length];
        for(int j=0;j<mid.length;j++) {

          mid[j] = fft_window[j]*y_frams[j][i];

        }

        double[] temp_real = fourier_real(mid);
        double[] temp_imag = fourier_imag(mid);
        for(int j=0;j<temp_imag.length;j++) {

          stft_matrix_real[j][i] = temp_real[j];
          stft_matrix_imag[j][i] = temp_imag[j];
        }
      }

    }

    for(int i=0;i<stft_matrix_real.length; i++) {
      for(int j=0;j<stft_matrix_real[0].length;j++) {
        stft_matrix_abs[i][j]=Math.sqrt(stft_matrix_real[i][j]*stft_matrix_real[i][j]+stft_matrix_imag[i][j]*stft_matrix_imag[i][j]);
      }
    }
    stft_matrix_abs = amplitude_to_db(stft_matrix_abs);
    return stft_matrix_abs;

  }

  public static double[] fourier_real(double[] data) {
    int N = data.length;
    double[] real = new double[N];
    for (int k = 0; k < N; k++) {
      double re = 0;

      for (int n = 0; n < N; n++) {
        re += data[n] * Math.cos((2 * Math.PI * k * n) / N);
      }
      real[k] = re;

    }
    double[] result = new double[(int)Math.floor(1+N/2)];
    for(int i=0; i<result.length; i++) {
      result[i] = real[i];
    }
    return result;
  }

  public static double[] fourier_imag(double[] data) {
    int N = data.length;
    double[] imag;
    imag = new double[N];
    for (int k = 0; k < N; k++) {
      double im = 0;
      for (int n = 0; n < N; n++) {
        im -= data[n] * Math.sin((2 * Math.PI * k * n) / N);
      }
      imag[k] = im;

    }
    double[] result = new double[(int)Math.floor(1+N/2)];
    for(int i=0; i<result.length; i++) {
      result[i] = imag[i];
    }
    return result;
  }

  public static double[] get_window(int length) {
    // M, needs_trunc = _extend(M, sym)
    length = length + 1;

    // fac = np.linspace(-np.pi, np.pi, M)
    // w = np.zeros(M)
    double[] fac = new double[length];
    double[] w = new double[length - 1];

    for (int i = 0; i < length - 1; i++) {
      w[i] = 0;
      fac[i] = (double) (-Math.PI + (i * 2 * Math.PI) / (length - 1));

    }
    fac[length - 1] = (double) (-Math.PI + 2 * Math.PI);
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < length - 1; j++) {
        w[j] = (double) (w[j] + 0.5 * Math.cos(i * fac[j]));

      }
    }

    return w;

  }

  public static double[][] reshape(double[] data) {
    double[][] result = new double[data.length][1];
    for (int i = 0; i < result.length; i++) {
      result[i][0] = data[i];
    }
    return result;
  }

  public static double[] center(double[] data, int pad) {

    double[] result = new double[data.length + pad * 2];
    for (int i = 0; i < pad; i++) {
      result[i] = data[pad - i];
    }
    for (int i = pad; i < pad + data.length; i++) {
      result[i] = data[i - pad];
    }
    for (int i = pad + data.length; i < 2 * pad + data.length; i++) {
      result[i] = data[data.length - 1 - (i - pad - data.length + 1)];
    }

    return result;
  }

  public static double[][] frame(double[] x, int frame_length, int hop_length) {
    int n_fram = (int) (1 + Math.floor((x.length - frame_length) / hop_length));

    double[][] result = new double[frame_length][n_fram];
    for (int i = 0; i < n_fram; i++) {
      for (int j = 0; j < frame_length; j++) {
        result[j][i] = x[i * hop_length + j];
      }
    }
    return result;
  }

  /******************************************************************************************/
  /***********************************power_to_db**********************************************/
  public static double[][] amplitude_to_db(double[][] magnitude) {
    //power = np.square(magnitude, out=magnitude)
    double[][] power = new double[magnitude.length][magnitude[0].length];
    double ref = Double.NEGATIVE_INFINITY;
    for(int i=0;i<magnitude.length;i++) {
      for(int j=0;j<magnitude[0].length;j++) {
        if(magnitude[i][j]>ref) ref = magnitude[i][j];
        power[i][j] = magnitude[i][j]*magnitude[i][j];
      }
    }
    ref = ref*ref;
    double amin = 1e-5*1e-5;
    double top_db = 80;
    double[][] mid = power_to_db(power,ref,amin,top_db);
    return resize(mid);
  }

  public static double[][] power_to_db(double[][] power, double ref, double amin, double top_db) {
    double[][] log_spec = new double[power.length][power[0].length];
    double max = Double.NEGATIVE_INFINITY;
    for(int i=0; i<power.length; i++) {
      for(int j=0; j<power[0].length; j++) {
        double mid1 = Math.max(amin,power[i][j]);
        double mid2 = Math.max(amin, ref);
        log_spec[i][j] = 10*Math.log10(mid1);
        log_spec[i][j]-= 10*Math.log10(mid2);
        if(log_spec[i][j]>max) max = log_spec[i][j];
      }
    }
    for(int i=0; i<power.length; i++) {
      for(int j=0; j<power[0].length; j++) {
        log_spec[i][j] = Math.max(log_spec[i][j], max-top_db);
      }
    }
    return log_spec;
  }
  /******************************************************************************************/
  /**************************************缩放图片*************************************************/
  public static double[][] resize(double[][] sig){
    double[][] result = new double[128][128];
    double max = Double.NEGATIVE_INFINITY;
    double min = Double.POSITIVE_INFINITY;
    for(int i=0;i<128;i++) {
      for(int j=0;j<128;j++) {
        int src_i = Math.round(i*(sig.length-1)/127);
        int src_j = Math.round(j*(sig[0].length-1)/127);
        result[i][j] = sig[src_i][src_j];
        if(result[i][j]>max) max = result[i][j];
        if(result[i][j]<min) min = result[i][j];
      }
    }
    for(int i=0;i<128;i++) {
      for(int j=0;j<128;j++) {
        result[i][j] = (result[i][j]-min)/(max-min);
      }
    }
    return result;
  }
  /********************************************************************************************/
  public void test(String path){
    Butterworth butterworth = new Butterworth();
    double[] actual = WaveReader(path);
    int fs = fs(path);
    int window_len;
    if(fs==44100) window_len = 10010;
    else if(fs==4000) window_len = 908;
    else if(fs==22050) window_len = 5000;
    else window_len = 1800;

    butterworth.bandPass(5, fs,1025 , 1950);
    List<Double> sig = new ArrayList<Double>();
      for (int i = 0; i < actual.length; i++) {
        double v = 0;
        v = butterworth.filter(actual[i]);
        sig.add(v);

      }

    List<Double> seperate = find_point(sig,fs, window_len);
    run_model(sig, seperate, fs);
  }


  @Override
  public synchronized void onStart() {
    LOGGER.d("onStart " + this);
    super.onStart();
  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }

  @Override
  public synchronized void onPause() {
    LOGGER.d("onPause " + this);

    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }

    super.onPause();
  }

  @Override
  public synchronized void onStop() {
    LOGGER.d("onStop " + this);
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    LOGGER.d("onDestroy " + this);
    super.onDestroy();
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }


  // Returns true if the device supports the required hardware level, or better.
  private boolean isHardwareLevelSupported(
      CameraCharacteristics characteristics, int requiredLevel) {
    int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
    if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
      return requiredLevel == deviceLevel;
    }
    // deviceLevel is not LEGACY, can use numerical sort
    return requiredLevel <= deviceLevel;
  }


  protected void setFragment() {
//    String cameraId = chooseCamera();

    Fragment fragment;

    CameraConnectionFragment camera2Fragment =
            CameraConnectionFragment.newInstance(

                    getLayoutId(),
                    getDesiredPreviewFrameSize());


    fragment = camera2Fragment;
    camera2Fragment.getResult(new interfaceCallback(){
      @Override
      public void getResult(String test){
        Toast.makeText(CameraActivity.this, test, Toast.LENGTH_LONG).show();
      }

    });
//    getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
  }

  /******************读入wav********************************/
  public static double[] WaveReader(String fileName){
    double[] actual = null;
    byte data[];
    try {
      data = readWave(fileName);
      actual = fileDescript(data); //set wave file header

    } catch (FileNotFoundException e) {
      LOGGER.e("File Not Found!!");
      e.printStackTrace();
    }
    return actual;
  }

  public static int fs(String fileName){
    byte data[];
    int fs=0;
    try {
      data = readWave(fileName);
      fs = read_fs(data); //set wave file header

    } catch (FileNotFoundException e) {
      LOGGER.e("File Not Found!!");
      e.printStackTrace();
    }
    return fs;
  }

  private static double[] fileDescript(byte[] data){
    int subChunk2Size;
    double actualData[];
    byte[] temp = new byte [] {data[40], data[41], data[42], data[43]};
    subChunk2Size = bigToLittleInt(temp);

    //Read sample part
    int sIndex=44, index=0;
    actualData = new double[subChunk2Size/2];
    while(sIndex<subChunk2Size+44){
      temp = new byte [] {data[sIndex++], data[sIndex++]};
      actualData[index] = bigToLittleDouble(temp);

      index++;

    }
    /* System.out.print("index:"+index+"\n"); */

    /*
     * for(int i=0;i<actualData.length;i++) { System.out.print(actualData[i]+"\n");
     * }
     */
    return actualData;

  }

  private static int read_fs(byte[] data){
    int sampleRate;

    byte[] temp = new byte [] {data[24], data[25], data[26], data[27]};
    sampleRate = bigToLittleInt(temp);

    return sampleRate;

  }
  private static double bigToLittleDouble(byte[] raw){
    for(int i = 0; i < raw.length / 2; i++)
    {
      byte temp = raw[i];
      raw[i] = raw[raw.length - i - 1];
      raw[raw.length - i - 1] = temp;
    }
    return (double) new BigInteger(raw).intValue();
  }

  private static int bigToLittleInt(byte[] raw){ //reverse array proved..
    for(int i = 0; i < raw.length / 2; i++)
    {
      byte temp = raw[i];
      raw[i] = raw[raw.length - i - 1];
      raw[raw.length - i - 1] = temp;
    }
    return byteArrayToInt(raw);
  }

  private static int byteArrayToInt(byte[] b) {
			/*for(byte r:b)
				System.out.printf("%02X ",r);*/
    //System.out.println();
    if (b.length == 4)
      return b[0] << 24 | (b[1] & 0xff) << 16 | (b[2] & 0xff) << 8
              | (b[3] & 0xff);
    else if (b.length == 2)
      return 0x00 << 24 | 0x00 << 16 | (b[0] & 0xff) << 8 | (b[1] & 0xff);

    return 0;
  }

  private static byte[] readWave(String path) throws FileNotFoundException{
    FileInputStream fin;
    int len;
    byte data[] = new byte[1000000];
    try {
      File file = new File(path);

      fin = new FileInputStream(file);

      do {
        len = fin.read(data);
		      /*for (int j = 0; j < len; j++)
		        System.out.printf("%02X ", data[j]);*/
      } while (len != -1);
//		    System.out.println("\nReading finish...");
      fin.close();
    } catch (IOException e) {
      e.printStackTrace();
    }

    return data;
  }

  private void read_wav(){
    String sdCardDir = Environment.getExternalStorageDirectory().getPath();
    File file = new File(sdCardDir+"/LBStethoscope/cache/bin/lbstethoscope_1703/recorded/lbstethoscope_1703_1637/");
    File[] files = file.listFiles();
    File[] wavs = files[0].listFiles();
    int num = 3;
    for(File wav : wavs){
      String fileName = wav.getName();
      TextView textview=new TextView(this);
      textview.setText("Record "+String.valueOf(num)+": "+String.valueOf(wav));
      num++;
      textview.setTextSize(15);
      textview.setTextColor(Color.BLACK);
      textview.setGravity(Gravity.CENTER);
      textview.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
          Normal_number.setText("-");
          Wheeze_number.setText("-");
          Crackle_number.setText("-");
          Both_number.setText("-");
          progressDialog.show();
          new Thread(new Runnable() {
            public void run() {
              test(String.valueOf(wav));
            }
          }).start();
        }
      });
      mainLinerLayout.addView(textview);
    }
  }
  /**************************************************/

//  @Override
//  public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
///*    setUseNNAPI(isChecked);
//    if (isChecked) apiSwitchCompat.setText("NNAPI");
//    else apiSwitchCompat.setText("TFLITE");*/
//  }

  @Override
  public void onClick(View v) {
    if (v.getId() == R.id.record1) {
      Normal_number.setText(String.valueOf(1)+" cycles");
      Wheeze_number.setText(String.valueOf(0)+" cycles");
      progressDialog.show();
      new Thread(new Runnable() {
        public void run() {
          String sdCardDir = Environment.getExternalStorageDirectory().getPath();
          test(sdCardDir+"/Sounds/5.wav");
        }
      }).start();

    } else if (v.getId() == R.id.record2) {
      Normal_number.setText(String.valueOf(9)+" cycles");
      Wheeze_number.setText(String.valueOf(0)+" cycles");
      progressDialog.show();
      new Thread(new Runnable() {
        public void run() {
          String sdCardDir = Environment.getExternalStorageDirectory().getPath();
          test(sdCardDir+"/Sounds/109_1b1_Al_sc_Litt3200.wav");
        }
      }).start();
    } else if (v.getId() == R.id.record3) {
      progressDialog.show();
      new Thread(new Runnable() {
        public void run() {
          String sdCardDir = Environment.getExternalStorageDirectory().getPath();
          test(sdCardDir+"/Sounds/222_1b1_Pr_sc_Meditron.wav");
        }
      }).start();

      Normal_number.setText(String.valueOf(7)+" cycles");
      Wheeze_number.setText(String.valueOf(8)+" cycles");
    }
  }

  protected void showFrameInfo(String frameInfo) {
    frameValueTextView.setText(frameInfo);
  }

  protected void showCropInfo(String cropInfo) {
    cropValueTextView.setText(cropInfo);
  }

  protected void showInference(String inferenceTime) {
    inferenceTimeTextView.setText(inferenceTime);
  }

  protected abstract void onPreviewSizeChosen();

  protected abstract int processImage(double[][] stft, double[][] mfcc);

//  protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

  protected abstract int getLayoutId();

  protected abstract Size getDesiredPreviewFrameSize();

  protected abstract void setNumThreads(int numThreads);

  protected abstract void setUseNNAPI(boolean isChecked);
}

package com.example.sensorcollector;

import android.app.Activity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Environment;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;


public class SensorDataManager implements SensorEventListener {
    private final int SAMPLING_RATE = 100000;

    private SensorManager mSensorManager;
    private boolean isCollecting;
    private BufferedWriter writer;

    public SensorDataManager(Context context) {
        mSensorManager = (android.hardware.SensorManager) context.getSystemService(Activity.SENSOR_SERVICE);
        isCollecting = false;
    }

    public void startCollection() {
        if (!isCollecting) {
            mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SAMPLING_RATE);
            mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE), SAMPLING_RATE);
            mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_ORIENTATION), SAMPLING_RATE);
            mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD), SAMPLING_RATE);
            mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_PRESSURE), SAMPLING_RATE);
            mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY), SAMPLING_RATE);
            mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION), SAMPLING_RATE);
            mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR), SAMPLING_RATE);
            isCollecting = true;

            try {
                File file = new File(Environment.getExternalStorageDirectory() + "/SensorCollector/" + System.currentTimeMillis() + ".txt");
                writer = new BufferedWriter(new FileWriter(file, true));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public void stopCollection() {
        if (isCollecting) {
            mSensorManager.unregisterListener(this);
            isCollecting = false;

            try {
                writer.flush();
                writer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        JSONObject data = new JSONObject();
        try {
            data.put("timestamp", System.currentTimeMillis());
            data.put("accuracy", event.accuracy);
            data.put("type", event.sensor.getType());
            JSONArray v = new JSONArray();
            for (float val : event.values) {
                v.put(val);
            }
            data.put("values", v);

            try {
                writer.write(data.toString());
                Log.d("haha", data.toString());
            } catch (Exception e) {
                e.printStackTrace();
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
    }
}
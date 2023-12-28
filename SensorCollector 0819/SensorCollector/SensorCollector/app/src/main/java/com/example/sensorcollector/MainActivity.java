package com.example.sensorcollector;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.Button;

import java.io.File;
import java.util.LinkedList;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private Button btnStart, btnStop;
    private SensorDataManager mSensorDataManager;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnStart = (Button) findViewById(R.id.button);
        btnStop = (Button) findViewById(R.id.button2);

        initViews();
        initVars();
        checkAndRequestPermission();

        File file = new File(Environment.getExternalStorageDirectory() + "/SensorCollector");
        if (!file.exists() || !file.isDirectory()) {
            file.mkdir();
        }
    }

    private void initVars(){
        mSensorDataManager = new SensorDataManager(this);
    }

    private void initViews() {
        btnStart.setOnClickListener(this);
        btnStop.setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.button:
                mSensorDataManager.startCollection();
                break;
            case R.id.button2:
                mSensorDataManager.stopCollection();
                break;
        }
    }

    private void checkAndRequestPermission() {
        LinkedList<String> requestedPermissions = new LinkedList<>();
        String[] allPermissions = new String[0];
        try {
            allPermissions = getPackageManager()
                    .getPackageInfo(this.getPackageName(), PackageManager.GET_PERMISSIONS)
                    .requestedPermissions;
        } catch (PackageManager.NameNotFoundException e) {
            e.printStackTrace();
        }
        for (String permission : allPermissions) {
            if (
                    ContextCompat.checkSelfPermission(this, permission)
                            != PackageManager.PERMISSION_GRANTED
            ) {
                requestedPermissions.add(permission);
            }
        }

        if (requestedPermissions.size() == 0) {
            return;
        }

        ActivityCompat.requestPermissions(
                this,
                requestedPermissions.toArray(new String[0]),
                1000
        );
    }

}

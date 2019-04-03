package com.sum10.cap_vino;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;

public class SplashActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstance) {
        super.onCreate(savedInstance);

        try {
            Thread.sleep(2000); // 2초 동안 로고화면 표시
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        startActivity(new Intent(this, MainActivity.class)); // 2초 경과 후 메인화면(테마 리스트)으로 이동
        finish(); // 로고화면 액티비티 종료
    }
}

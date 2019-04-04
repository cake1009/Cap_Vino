package com.sum10.cap_vino;

import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.design.widget.BottomNavigationView;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentTransaction;
import android.support.v7.app.AppCompatActivity;
import android.view.MenuItem;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    private TextView mTextMessage;
    private FragmentManager fragmentManager = getSupportFragmentManager();

    private Menu1Fragment menu1Fragment = new Menu1Fragment();
    private Menu2Fragment menu2Fragment = new Menu2Fragment();
    private Menu3Fragment menu3Fragment = new Menu3Fragment();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        FragmentTransaction transaction = fragmentManager.beginTransaction();
        transaction.replace(R.id.framelayout, menu1Fragment).commitAllowingStateLoss();

        mTextMessage = (TextView) findViewById(R.id.message);
        BottomNavigationView navigation = (BottomNavigationView) findViewById(R.id.navigation);

        navigation.setOnNavigationItemSelectedListener(new BottomNavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
                FragmentTransaction transaction = fragmentManager.beginTransaction();

                switch(menuItem.getItemId()) {
                    case R.id.navigation_home :
                        transaction.replace(R.id.framelayout, menu1Fragment).commitAllowingStateLoss();
                        break;

                    case R.id.navigation_camera :
                        transaction.replace(R.id.framelayout, menu2Fragment).commitAllowingStateLoss();
                        break;

                    case R.id.navigation_mypage :
                        transaction.replace(R.id.framelayout, menu3Fragment).commitAllowingStateLoss();
                        break;
                }
                return true;
            }
        });
    }

}

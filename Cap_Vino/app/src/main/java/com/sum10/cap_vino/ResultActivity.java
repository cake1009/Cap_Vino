package com.sum10.cap_vino;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;

import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;

public class ResultActivity extends AppCompatActivity {

    private Intent intent;
    private String output;
    private TextView textView;
    private ImageView imageView;
    private String path;
    private Bitmap bitmap;

    private StorageReference storageReference;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        path = "a";

        storageReference = FirebaseStorage.getInstance().getReferenceFromUrl("gs://cap-vino.appspot.com");

        textView = findViewById(R.id.textView2);
        imageView = findViewById(R.id.imageView2);

        intent = getIntent();
        output = intent.getStringExtra("output");
        storageReference.child("output").child(output + ".jpg").getDownloadUrl().addOnSuccessListener(new OnSuccessListener<Uri>() {
            @Override
            public void onSuccess(final Uri uri) {
                Thread thread = new Thread() {
                  @Override
                  public void run() {
                      try {
                          URL url = new URL(uri.toString());
                          URLConnection connection = url.openConnection();
                          connection.connect();

                          InputStream is = connection.getInputStream();
                          bitmap = BitmapFactory.decodeStream(is);

                      } catch(Exception e) { }
                  }
                };
                thread.start();

                try {
                    thread.join();
                    imageView.setImageBitmap(bitmap);
                } catch(InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {
                Log.d("tag", "이미지가 없습니다");
            }
        });

        textView.setText(output);
    }
}

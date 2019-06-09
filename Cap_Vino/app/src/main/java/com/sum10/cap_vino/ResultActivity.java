package com.sum10.cap_vino;

import android.content.Intent;
import android.net.Uri;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;

import java.net.URL;
import java.net.URLConnection;

public class ResultActivity extends AppCompatActivity {

    private Intent intent;
    private String output;
    private TextView textView;
    private String path;

    private StorageReference storageReference;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result);

        path = "a";

        storageReference = FirebaseStorage.getInstance().getReferenceFromUrl("gs://cap-vino.appspot.com");

        textView = findViewById(R.id.textView2);

        intent = getIntent();
        output = intent.getStringExtra("output");
        storageReference.child("images").child(output + ".jpg").getDownloadUrl().addOnSuccessListener(new OnSuccessListener<Uri>() {
            @Override
            public void onSuccess(Uri uri) {
                path = uri.toString();
            }
        }).addOnFailureListener(new OnFailureListener() {
            @Override
            public void onFailure(@NonNull Exception e) {

            }
        });

        textView.setText(path);
    }
}

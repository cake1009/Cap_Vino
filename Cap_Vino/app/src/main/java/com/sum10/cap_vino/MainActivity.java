package com.sum10.cap_vino;

import android.Manifest;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.android.gms.auth.api.signin.GoogleSignIn;
import com.google.android.gms.auth.api.signin.GoogleSignInAccount;
import com.google.android.gms.auth.api.signin.GoogleSignInClient;
import com.google.android.gms.auth.api.signin.GoogleSignInOptions;
import com.google.android.gms.common.SignInButton;
import com.google.android.gms.common.api.ApiException;
import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthCredential;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.auth.GoogleAuthProvider;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainActivity extends AppCompatActivity {

    private static final int RC_SIGN_IN = 10;
    private final int CAMERA_CODE = 1112; //갤러리 접근 권한 코드
    private final int REQUEST_PERMISSION_CODE = 2222; //퍼미션 요청 코드
    private ImageView imageView;
    private String mCurrentPhotoPath;
    private StorageReference storageReference;
    private GoogleSignInClient mGoogleSignInClient;
    private FirebaseAuth mAuth;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        mAuth = FirebaseAuth.getInstance();

        // Configure Google Sign In
        GoogleSignInOptions gso = new GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
                .requestIdToken(getString(R.string.default_web_client_id))
                .requestEmail()
                .build();

        mGoogleSignInClient = GoogleSignIn.getClient(this, gso);

        SignInButton button = (SignInButton)findViewById(R.id.login_button) ; //로그인버튼
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent signInIntent = mGoogleSignInClient.getSignInIntent();
                startActivityForResult(signInIntent, RC_SIGN_IN);
                //System.out.println("클릭 테스트");
            }
        });

        storageReference = FirebaseStorage.getInstance().getReference();
        imageView = findViewById(R.id.imageView);

        FloatingActionButton fab = findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int permission_camera = ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA);
                int permission_writeExternalStorage = ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE);

                if (permission_camera == PackageManager.PERMISSION_GRANTED && permission_writeExternalStorage == PackageManager.PERMISSION_GRANTED)
                    onCamera();
                else {
                    requestPermission();
                    if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this, Manifest.permission.CAMERA)
                            || ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this, Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                        //거부했을 경우
                        Toast.makeText(MainActivity.this, "기능 사용을 위한 권한 동의가 필요합니다.", Toast.LENGTH_SHORT).show();
                    }
                }
            }
        });
    }
    //명순철123123
    private void onCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        if (intent.resolveActivity(getPackageManager()) != null) {
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException e) {}

            if (photoFile != null) {
                Uri photoURI = FileProvider.getUriForFile(MainActivity.this, getPackageName(), photoFile);
                intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(intent, CAMERA_CODE);
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);


        // Result returned from launching the Intent from GoogleSignInApi.getSignInIntent(...);
        if (requestCode == RC_SIGN_IN) {
            Task<GoogleSignInAccount> task = GoogleSignIn.getSignedInAccountFromIntent(data);
            try {
                // Google Sign In was successful, authenticate with Firebase
                GoogleSignInAccount account = task.getResult(ApiException.class);
                firebaseAuthWithGoogle(account);
            } catch (ApiException e) {
                // Google Sign In failed, update UI appropriately
               // Log.w(TAG, "Google sign in failed", e);
                // ...
            }
        }

        ExifInterface exifInterface = null;
        try {
            if (resultCode == RESULT_OK) {
                switch (requestCode) {
                    case CAMERA_CODE:
                        File file = new File(mCurrentPhotoPath);
                        Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), Uri.fromFile(file));

                        try {
                            exifInterface = new ExifInterface(mCurrentPhotoPath);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        assert exifInterface != null;
                        int exifOrientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
                        int exifDegree = exifOrientationToDegrees(exifOrientation);

                        if (bitmap != null) {
                            //imageView.setImageBitmap(rotate(bitmap, exifDegree)); //이미지뷰에 세팅
                            AlertDialog.Builder alert_image = new AlertDialog.Builder(MainActivity.this);
                            alert_image.setTitle("이 사진으로 검색하십니까?");
                            ImageView dialogImage = new ImageView(MainActivity.this);
                            dialogImage.setImageBitmap(rotate(bitmap, exifDegree));
                            alert_image.setView(dialogImage);

                            alert_image.setPositiveButton("검색", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialogInterface, int i) {
                                    Uri uri = Uri.fromFile(new File(mCurrentPhotoPath));
                                    StorageReference imageReference = storageReference.child("images/" + uri.getLastPathSegment());

                                    imageReference.putFile(uri)
                                            .addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
                                                @Override
                                                public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                                                    // Get a URL to the uploaded content
                                                    //Uri downloadUrl = taskSnapshot.getDownloadUrl();
                                                }
                                            })
                                            .addOnFailureListener(new OnFailureListener() {
                                                @Override
                                                public void onFailure(@NonNull Exception exception) {
                                                    // Handle unsuccessful uploads
                                                    // ...
                                                }
                                            });
                                    Intent intent = new Intent(MainActivity.this, LoadingActivity.class);
                                    startActivity(intent);
                                }
                            });
                            alert_image.show();
                        }
                        break;
                    default:
                        break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //갤러리 관련 메소드
    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(imageFileName,".jpg", storageDir);

        // Save a file: path for use with ACTION_VIEW intents
        mCurrentPhotoPath = image.getAbsolutePath();
        return image;
    }

    private int exifOrientationToDegrees(int exifOrientation) { //이미지의 절대경로(각도)를 가져옴
        if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_90)
            return 90;
        else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_180)
            return 180;
        else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_270)
            return 270;
        return 0;
    }

    private Bitmap rotate(Bitmap src, float degree) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degree);
        return Bitmap.createBitmap(src, 0, 0, src.getWidth(), src.getHeight(), matrix, true);
    }

    private void requestPermission() {
        ActivityCompat.requestPermissions(this, new String[]{ Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_PERMISSION_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case REQUEST_PERMISSION_CODE:
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED)
                    onCamera();
                else {
                    Toast.makeText(this, "기능 사용을 위한 권한 동의가 필요합니다.", Toast.LENGTH_SHORT).show();
                }
                break;
        }
    }

    private void firebaseAuthWithGoogle(GoogleSignInAccount acct) {

        AuthCredential credential = GoogleAuthProvider.getCredential(acct.getIdToken(), null);
        mAuth.signInWithCredential(credential)
                .addOnCompleteListener(this, new OnCompleteListener<AuthResult>() {
                    @Override
                    public void onComplete(@NonNull Task<AuthResult> task) {
                        if (!task.isSuccessful()) {
                            // Sign in success, update UI with the signed-in user's information
                         //   Log.d(TAG, "signInWithCredential:success");
                         //   FirebaseUser user = mAuth.getCurrentUser();
                        //    updateUI(user);
                        } else {

                            Toast.makeText(MainActivity.this, "FireBase 아이디 생성 완료되었습니다.",Toast.LENGTH_SHORT).show();
                            // If sign in fails, display a message to the user.
                         //   Log.w(TAG, "signInWithCredential:failure", task.getException());
                         //   Snackbar.make(findViewById(R.id.main_layout), "Authentication Failed.", Snackbar.LENGTH_SHORT).show();
                          //  updateUI(null);
                        }

                        // ...
                    }
                });
    }
}

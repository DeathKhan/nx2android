package org.nx2a;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

/**
 * Keeps a game that was left with HOME in memory. A paused game is a cached process of several GB: Android's low memory killer takes it
 * first, and the cached-app freezer ends it when its binder buffer fills ("excessive binder traffic during cached"), so coming back
 * started the game again. A foreground service keeps the process out of the cached state. Started by front_android.c when the APK has it
 * (mkapk.py --keep-alive); the notification shows only if the player allows notifications for the game.
 */
public class KeepAlive extends Service {
    @Override public IBinder onBind(Intent intent) { return null; }

    @Override public int onStartCommand(Intent intent, int flags, int id) {
        try {
            getSystemService(NotificationManager.class).createNotificationChannel(new NotificationChannel("game", "Game in memory", NotificationManager.IMPORTANCE_LOW));
            Notification n = new Notification.Builder(this, "game").setContentTitle(getApplicationInfo().loadLabel(getPackageManager()))
                .setContentText("Kept in memory while in the background").setSmallIcon(android.R.drawable.ic_media_pause).setOngoing(true).build();
            startForeground(1, n, 2);   // ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK
        } catch (RuntimeException e) {   // not allowed (started from the background): the game runs as before, without the protection
            Log.e("dmhost", "keep-alive service", e); stopSelf();
        }
        return START_NOT_STICKY;
    }

    @Override public void onTaskRemoved(Intent root) { stopSelf(); }   // swiped off Recents: the game ends (front_android.c on_destroy)
}

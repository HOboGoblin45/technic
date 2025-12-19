import Flutter
import UIKit
import BackgroundTasks
import FirebaseCore
import FirebaseMessaging

@main
@objc class AppDelegate: FlutterAppDelegate {

    /// Background task identifier for alert checking
    static let alertCheckTaskId = "com.technic.technicMobile.alertCheck"

    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        // Initialize Firebase
        FirebaseApp.configure()

        // Set up Firebase Messaging delegate
        Messaging.messaging().delegate = self

        // Request notification permissions
        UNUserNotificationCenter.current().delegate = self
        let authOptions: UNAuthorizationOptions = [.alert, .badge, .sound]
        UNUserNotificationCenter.current().requestAuthorization(
            options: authOptions,
            completionHandler: { _, _ in }
        )
        application.registerForRemoteNotifications()

        GeneratedPluginRegistrant.register(with: self)

        // Register background tasks (iOS 13+)
        if #available(iOS 13.0, *) {
            registerBackgroundTasks()
        }

        // Enable background fetch (legacy support)
        application.setMinimumBackgroundFetchInterval(
            UIApplication.backgroundFetchIntervalMinimum
        )

        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }

    // MARK: - Background Tasks (iOS 13+)

    @available(iOS 13.0, *)
    private func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: AppDelegate.alertCheckTaskId,
            using: nil
        ) { task in
            self.handleAlertCheckTask(task: task as! BGAppRefreshTask)
        }

        print("[AppDelegate] Background task registered: \(AppDelegate.alertCheckTaskId)")
    }

    @available(iOS 13.0, *)
    private func handleAlertCheckTask(task: BGAppRefreshTask) {
        // Schedule the next background task
        scheduleAlertCheckTask()

        // Create a task expiration handler
        task.expirationHandler = {
            task.setTaskCompleted(success: false)
        }

        // Perform the alert check via Flutter method channel
        // The actual check is done by Flutter code
        if let controller = window?.rootViewController as? FlutterViewController {
            let channel = FlutterMethodChannel(
                name: "com.technic.technicMobile/alerts",
                binaryMessenger: controller.binaryMessenger
            )

            channel.invokeMethod("checkAlerts", arguments: nil) { result in
                task.setTaskCompleted(success: true)
            }
        } else {
            task.setTaskCompleted(success: true)
        }
    }

    /// Schedule the next background alert check
    @available(iOS 13.0, *)
    func scheduleAlertCheckTask() {
        let request = BGAppRefreshTaskRequest(identifier: AppDelegate.alertCheckTaskId)
        // Schedule for 15 minutes from now (minimum iOS allows)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60)

        do {
            try BGTaskScheduler.shared.submit(request)
            print("[AppDelegate] Scheduled next alert check")
        } catch {
            print("[AppDelegate] Failed to schedule alert check: \(error)")
        }
    }

    // MARK: - Legacy Background Fetch (iOS 12 and earlier)

    override func application(
        _ application: UIApplication,
        performFetchWithCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void
    ) {
        // This is called for legacy background fetch
        // Forward to Flutter if needed
        if let controller = window?.rootViewController as? FlutterViewController {
            let channel = FlutterMethodChannel(
                name: "com.technic.technicMobile/alerts",
                binaryMessenger: controller.binaryMessenger
            )

            channel.invokeMethod("checkAlerts", arguments: nil) { result in
                completionHandler(.newData)
            }
        } else {
            completionHandler(.noData)
        }
    }

    // MARK: - App Lifecycle

    override func applicationDidEnterBackground(_ application: UIApplication) {
        // Schedule background task when app enters background
        if #available(iOS 13.0, *) {
            scheduleAlertCheckTask()
        }
    }

    // MARK: - Remote Notifications

    override func application(
        _ application: UIApplication,
        didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data
    ) {
        // Pass device token to Firebase
        Messaging.messaging().apnsToken = deviceToken
        print("[AppDelegate] APNs token registered")
    }

    override func application(
        _ application: UIApplication,
        didFailToRegisterForRemoteNotificationsWithError error: Error
    ) {
        print("[AppDelegate] Failed to register for remote notifications: \(error)")
    }
}

// MARK: - Firebase Messaging Delegate

extension AppDelegate: MessagingDelegate {
    func messaging(_ messaging: Messaging, didReceiveRegistrationToken fcmToken: String?) {
        print("[AppDelegate] FCM Token: \(fcmToken ?? "nil")")

        // Send token to your backend server if needed
        let dataDict: [String: String] = ["token": fcmToken ?? ""]
        NotificationCenter.default.post(
            name: Notification.Name("FCMToken"),
            object: nil,
            userInfo: dataDict
        )

        // Forward to Flutter via method channel if needed
        if let controller = window?.rootViewController as? FlutterViewController {
            let channel = FlutterMethodChannel(
                name: "com.technic.technicMobile/fcm",
                binaryMessenger: controller.binaryMessenger
            )
            channel.invokeMethod("onTokenRefresh", arguments: fcmToken)
        }
    }
}

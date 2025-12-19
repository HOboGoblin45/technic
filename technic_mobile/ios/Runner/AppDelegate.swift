import Flutter
import UIKit
import BackgroundTasks

@main
@objc class AppDelegate: FlutterAppDelegate {

    /// Background task identifier for alert checking
    static let alertCheckTaskId = "com.technic.technicMobile.alertCheck"

    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
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
}

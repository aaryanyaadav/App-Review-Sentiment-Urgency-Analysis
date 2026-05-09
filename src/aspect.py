class AspectDetector:
    def __init__(self):
        self.aspect_dict = {

            "UI": [
                "ui", "design", "interface", "layout", "theme",
                "color", "font", "navigation", "appearance",
                "look", "feel", "style", "visual", "ux",
                "user interface", "dark mode", "light mode",
                "display", "screen", "alignment", "spacing",
                "icons", "buttons", "menu", "dashboard",
                "home screen", "animation", "transition",
                "responsiveness", "clarity", "readability",
                "modern design", "outdated design",
                "customization", "personalization",
                "visual bug", "bad design", "ugly",
                "clean design", "intuitive", "confusing ui"
            ],

            "Performance": [
                "slow", "lag", "lagging", "freeze", "crash",
                "crashing", "hang", "delay", "performance",
                "stutter", "fps drop", "unresponsive",
                "loading time", "takes time", "very slow",
                "speed", "optimization", "heavy app",
                "cpu usage", "memory usage", "overload",
                "buffering", "timeout", "not responding",
                "high latency", "low performance",
                "sluggish", "frame drop", "freeze screen",
                "stuck", "delay response", "slow startup",
                "app not smooth", "bad performance",
                "performance issue", "overload device",
                "high ram usage", "background lag"
            ],

            "Battery": [
                "battery", "drain", "power", "consumption",
                "heating", "overheat", "battery drain fast",
                "battery usage", "power usage", "energy",
                "battery issue", "phone heating",
                "device heating", "hot phone",
                "thermal issue", "battery hog",
                "high battery usage", "drains quickly",
                "battery problem", "battery performance",
                "low battery backup", "power drain",
                "overheating issue", "heat up",
                "warm device", "battery optimization",
                "battery life", "battery drop",
                "power consumption high", "heating problem",
                "phone gets hot", "device gets hot",
                "energy drain", "power hungry"
            ],

            "Network": [
                "network", "internet", "connection", "wifi",
                "disconnect", "offline", "signal",
                "no internet", "poor connection",
                "weak signal", "network error",
                "connection lost", "reconnect",
                "data issue", "mobile data",
                "wifi issue", "internet slow",
                "network slow", "unstable connection",
                "no signal", "airplane mode issue",
                "server unreachable", "network timeout",
                "connection timeout", "network drop",
                "packet loss", "latency issue",
                "dns issue", "network problem",
                "internet problem", "wifi disconnecting",
                "connection failed", "cannot connect"
            ],

            "Login/Auth": [
                "login", "signin", "sign in", "authentication",
                "otp", "password", "account", "verification",
                "login failed", "cannot login",
                "wrong password", "forgot password",
                "reset password", "account locked",
                "login issue", "auth error",
                "two factor", "2fa", "verification failed",
                "invalid credentials", "access denied",
                "session expired", "login timeout",
                "signin error", "account issue",
                "user id", "username", "login problem",
                "auth problem", "login bug",
                "cannot sign in", "otp not received",
                "verification issue", "account blocked",
                "security login issue"
            ],

            "Payment": [
                "payment", "refund", "money", "transaction",
                "upi", "bank", "failed payment", "billing",
                "charged twice", "double charge",
                "payment failed", "transaction failed",
                "refund not received", "refund delay",
                "payment issue", "billing issue",
                "upi failed", "bank error",
                "payment gateway", "checkout issue",
                "cannot pay", "payment declined",
                "amount deducted", "money deducted",
                "transaction error", "invoice issue",
                "subscription issue", "auto debit",
                "payment stuck", "pending payment",
                "billing problem", "wrong charge",
                "overcharged", "payment glitch"
            ],

            "Ads": [
                "ads", "advertisement", "ads pop",
                "too many ads", "intrusive ads",
                "pop up ads", "ads everywhere",
                "annoying ads", "ad spam",
                "video ads", "banner ads",
                "ads interrupt", "ads issue",
                "ads problem", "ads loading",
                "ads not closing", "ads blocking",
                "forced ads", "ads frequency",
                "ads overload", "ads bug",
                "unskippable ads", "ads popup",
                "ads disturbing", "ads lag",
                "ads slow app", "ads crash",
                "ads freeze", "ads glitch",
                "ads interrupting", "too much advertising"
            ],

            "Notification": [
                "notification", "alert", "reminder",
                "push notification", "no notification",
                "notification delay", "late notification",
                "notification not working",
                "missing notification", "spam notification",
                "too many notifications", "notification bug",
                "notification issue", "alert issue",
                "reminder not working",
                "push not working", "silent notification",
                "notification error", "notification glitch",
                "notification lag", "duplicate notification",
                "delayed alert", "notification problem",
                "no alerts", "notification spam",
                "notification settings issue"
            ],

            "Updates": [
                "update", "latest version", "new version",
                "upgrade", "patch", "update issue",
                "update failed", "cannot update",
                "update problem", "update bug",
                "new update", "recent update",
                "version issue", "update crash",
                "update lag", "update error",
                "update stuck", "update slow",
                "update broken", "after update",
                "post update issue", "update glitch",
                "update download failed",
                "install update", "update not working"
            ],

            "Compatibility": [
                "compatible", "device", "android", "ios",
                "version", "support", "not compatible",
                "device not supported", "os issue",
                "android version", "ios version",
                "compatibility issue", "unsupported device",
                "old device", "new device issue",
                "tablet issue", "phone issue",
                "resolution issue", "screen size issue",
                "device crash", "not working on device",
                "compatibility bug", "os compatibility",
                "hardware issue", "software mismatch"
            ],

            "Features": [
                "feature", "option", "function", "tool",
                "add feature", "missing feature",
                "feature request", "new feature",
                "need feature", "lack of feature",
                "feature issue", "feature bug",
                "functionality", "extra feature",
                "improve feature", "remove feature",
                "feature update", "better feature",
                "feature not working", "broken feature",
                "limited feature", "advanced feature",
                "feature improvement", "more options"
            ],

            "Security": [
                "security", "privacy", "data", "hack",
                "safe", "permission", "breach",
                "data leak", "data loss",
                "privacy issue", "security issue",
                "unauthorized access", "hacked account",
                "data breach", "malware", "virus",
                "phishing", "security risk",
                "encryption", "unsafe", "not secure",
                "data stolen", "account hacked",
                "permission issue", "security bug",
                "privacy concern", "data exposure"
            ],

            "Customer Support": [
                "support", "help", "customer care",
                "service", "response", "no reply",
                "no response", "slow support",
                "bad support", "poor service",
                "support issue", "help issue",
                "customer service", "no help",
                "support delay", "support problem",
                "no assistance", "unhelpful",
                "support not working", "support team",
                "contact support", "support bug",
                "no solution", "ignored request"
            ],

            "Download/Install": [
                "download", "install", "installation",
                "update failed", "not installing",
                "download failed", "install failed",
                "cannot download", "cannot install",
                "download issue", "install issue",
                "installation error", "download stuck",
                "install stuck", "download slow",
                "installation problem", "setup issue",
                "app not installing", "apk issue",
                "download error", "installation bug",
                "install crash", "download crash"
            ],

            "Bug": [
                "bug", "error", "issue", "problem",
                "glitch", "fault", "defect",
                "unexpected behavior", "broken",
                "not working", "fails", "failure",
                "system error", "runtime error",
                "logic error", "crash bug",
                "ui bug", "functional bug",
                "critical bug", "minor bug",
                "major issue", "technical issue",
                "random bug", "weird behavior",
                "abnormal behavior", "buggy",
                "unstable", "inconsistent",
                "app issue", "software bug"
            ]
        
        }

    def detect(self, text):
        text = text.lower()

        aspect_counts = {}

        for aspect, keywords in self.aspect_dict.items():
            count = 0

            for word in keywords:
                count += text.count(word)  

            if count > 0:
                aspect_counts[aspect] = count

        if not aspect_counts:
            return ["General Issue"]

        max_count = max(aspect_counts.values())

        dominant_aspects = [
            aspect for aspect, count in aspect_counts.items()
            if count == max_count
        ]

        return dominant_aspects
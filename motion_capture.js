/**
 * motion_capture.js
 * Integrates Google MediaPipe Pose for real-time joint tracking
 * and executes 3D skeletal retargeting mathematical operations using Three.js vectors.
 */

class MotionCaptureEngine {
    constructor(videoElement, avatarEngineInstance) {
        this.video = videoElement;
        this.avatar = avatarEngineInstance;
        
        if (!this.video) {
            console.error("MotionCaptureEngine initialization failed: Video missing!");
            return;
        }
        
        this.pose = null;
        this.cameraHelper = null;
        this.isTracking = false;

        // --- NEW: Pose-Triggered Performance Config (Disabled by default for new flow) ---
        this.isPoseTriggerMode = false; 
        this.activePoseKey = 'optimal_1';
        this.matchStartTime = null;
        this.requiredHoldTime = 1500;
        
        this.isModelReady = false; // Prevents race conditions with camera start
        
        // --- NEW: Interaction & Gesture Detection States ---
        this.spinState = 0;           // 0: Front, 1: Side, 2: Back, 3: Side 2
        this.spinTimer = 0;
        this.maxShoulderWidth = 0.15; // Rolling self-calibrating maximum
        this.baseOrder = null;        // Baseline left-to-right shoulder order
        
        this.prevLandmarks = {};      // For velocity/displacement tracking
        this.movementIntensity = 0.5; // Smooth rolling movement velocity
        this.stillStartTime = null;   // Timer for stillness
        this.isStill = false;         // Active state flag
        
        this.absenceStartTime = null; // Timer for user leaving frame
        this.isAbsent = true;         // Absence state flag (starts true so first detection triggers return)
        this.hasStarted = false;      // Track first valid frame
        this.isPaused = false;        // CPU Saver: pause pose inference when not in interaction states

        
        // Callbacks to communicate with index.html frontend
        this.onPoseScore = null;   
        this.onPoseSuccess = null; 

        // 🏺 乐舞胡旋——骨骼角度定义数据库
        this.targetPoses = {
            optimal_1: {
                name: "左手扬起弧度",
                description: "只需检测左手往上举起弯曲成弧度即可",
                angles: {
                    // 0 = straight up, 90 = horizontal, 180 = straight down.
                    // 60 degrees means arm raised diagonally upwards.
                    leftShoulder: 60 
                }
            },
            spread_arms: {
                name: "拉开双臂",
                description: "双手向两侧水平拉开",
                angles: {
                    leftShoulder: 90,
                    rightShoulder: 90
                }
            },
            optimal_2: {
                name: "大唐拂袖 · 抬起左手",
                description: "抬起左手高于肩膀即可",
                angles: {
                    leftShoulder: 140,
                    leftElbow: 165
                }
            },
            optimal_3: {
                name: "仙人指路 · 执乐仕女",
                description: "左手斜斜向外轻盈扬起，右手弯曲捧至胸前",
                angles: {
                    leftShoulder: 135, // Left arm raised away from waist
                    leftElbow: 165,    // Left elbow relaxed straight
                    rightShoulder: 135, // Right arm pointing down-forward
                    rightElbow: 90     // Right elbow bent in front of chest
                }
            },
            optimal_4: {
                name: "敦煌飞天 · 飞天乐伎",
                description: "左手高抬高于肩膀，右手向外微微舒展张开",
                angles: {
                    leftShoulder: 140, // Left arm raised high above shoulder
                    leftElbow: 165,    // Left elbow relaxed straight
                    rightShoulder: 150, // Right arm raised slightly away from body
                    rightElbow: 165    // Right elbow relaxed straight
                }
            }
        };
        
        this.initMediaPipe();
    }
    
    /**
     * Initializes Google MediaPipe Pose model
     */
    initMediaPipe() {
        try {
            console.log("[Diag - initMediaPipe] Step 1: before new Pose");
            this.pose = new Pose({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
                }
            });
            console.log(`[Diag - initMediaPipe] Step 2: after new Pose, type of this.pose: ${typeof this.pose}`);
            
            this.pose.setOptions({
                modelComplexity: 0, // Set to 0 (Lite) for much faster and more reliable initial detection
                smoothLandmarks: true,
                enableSegmentation: false,
                smoothSegmentation: false,
                minDetectionConfidence: 0.40, // More lenient
                minTrackingConfidence: 0.40
            });
            console.log("[Diag - initMediaPipe] Step 3: after setOptions");
            
            this.pose.onResults((results) => this.onPoseResults(results));
            console.log("[Diag - initMediaPipe] Step 4: after onResults");
            
            // Explicitly initialize to catch errors early
            this.pose.initialize().then(() => {
                console.log("[Mocap Engine] ✅ MediaPipe Pose model initialized successfully!");
                this.isModelReady = true;
            }).catch(e => {
                console.error("[Mocap Engine] ❌ MediaPipe Pose initialization FAILED:", e);
                if (typeof window.updateTrackingBadge === 'function') {
                    window.updateTrackingBadge("error", "AI加载失败");
                }
            });
            
            console.log("MediaPipe Pose model initialization started.");
        } catch (e) {
            console.error(`[Diag - initMediaPipe] CRASH! Error: ${e.message}`, e);
            throw e;
        }
    }
    
    /**
     * Starts the webcam capturing loop and feeds it to the Pose model
     */
    start() {
        console.log(`[Diag - Mocap.start] isTracking=${this.isTracking}, dimensions=${this.video.videoWidth}x${this.video.videoHeight}`);
        if (this.isTracking) return;
        
        // Safety Guard: Delay startup if the browser has not yet initialized the video track buffer (dimensions are 0x0)
        if (this.video.videoWidth === 0 || this.video.videoHeight === 0) {
            console.warn("[Mocap] Video element has 0x0 dimensions! Delaying camera helper startup by 300ms...");
            setTimeout(() => this.start(), 300);
            return;
        }
        
        console.log("[Mocap] Starting Camera loop...");
        
        this.isTracking = true;
        
        // Check if input is a physical webcam (has srcObject stream) or a playing video file (has src URL)
        const isWebcam = this.video.srcObject !== null;
        
        if (isWebcam) {
            console.log("Starting motion capture driven by physical Webcam...");
            try {
                this.cameraHelper = new Camera(this.video, {
                    onFrame: async () => {
                        if (this.isTracking && this.isModelReady && this.pose && !this.isPaused) {
                            if (!this.firstFrameSent) {
                                console.log("[Mocap Engine] ✅ First frame sent to MediaPipe Pose!");
                                this.firstFrameSent = true;
                            }
                            try {
                                await this.pose.send({ image: this.video });
                            } catch(e) {
                                console.warn("[Mocap Engine] Frame send error:", e);
                            }
                        }
                    },
                    width: 640,
                    height: 480
                });
                this.cameraHelper.start();
            } catch (e) {
                console.error("Error starting physical webcam capture:", e);
                this.isTracking = false;
            }
        } else {
            console.log("Starting motion capture driven by playing Video File...");
            // Local frame extraction loop using browser animation frames
            const frameLoop = async () => {
                if (!this.isTracking) return;
                
                if (!this.video.paused && !this.video.ended && this.isModelReady && this.pose) {
                    try {
                        if (typeof window.incrementFrameCount === 'function') {
                            window.incrementFrameCount();
                        }
                        await this.pose.send({ image: this.video });
                    } catch (e) {
                        console.error("Error sending video frame to MediaPipe Pose:", e);
                        if (this.isTracking) {
                            this.isTracking = false; // Stop the loop immediately to prevent alert spam
                            if (typeof showToast === 'function') {
                                showToast(`MediaPipe Pose error: ${e.message}`, "error");
                            } else {
                                console.error(`MediaPipe Pose error: ${e.message}`);
                            }
                        }
                    }
                }
                // Queue next frame loop
                if (this.isTracking) {
                    requestAnimationFrame(frameLoop);
                }
            };
            frameLoop();
        }
    }
    
    /**
     * Stops the webcam tracking loop
     */
    stop() {
        if (!this.isTracking) return;
        
        if (this.cameraHelper) {
            this.cameraHelper.stop();
        }
        this.isTracking = false;
        console.log("Webcam tracking stopped.");
    }
    
    pause() {
        console.log("[Mocap Engine] Pausing pose tracking (skipping inference)...");
        this.isPaused = true;
    }
    
    resume() {
        console.log("[Mocap Engine] Resuming pose tracking...");
        this.isPaused = false;
        this.stillStartTime = null;
        this.absenceStartTime = null;
    }
    
    /**
     * Evaluates whether key landmarks indicate a real human is present in the frame
     */
    isHumanPresent(results) {
        if (!this.logCounter) this.logCounter = 0;
        this.logCounter++;
        const shouldLog = (this.logCounter % 60 === 0);

        if (!results || !results.poseLandmarks) {
            if (shouldLog) console.log("[Mocap Debug] No poseLandmarks returned by MediaPipe");
            return false;
        }
        if (results.poseLandmarks.length < 29) {
            if (shouldLog) console.log("[Mocap Debug] poseLandmarks length too small:", results.poseLandmarks.length);
            return false;
        }
        const lm = results.poseLandmarks;
        if (!lm[11] || !lm[12] || !lm[0]) {
            if (shouldLog) console.log("[Mocap Debug] Missing key landmarks (0, 11, 12):", !!lm[0], !!lm[11], !!lm[12]);
            return false;
        }
        
        const keyIndices = [0, 11, 12, 23, 24]; // Nose, Shoulders and Hips
        let totalVis = 0;
        let count = 0;
        for (let idx of keyIndices) {
            if (lm[idx] && typeof lm[idx].visibility === 'number') {
                totalVis += lm[idx].visibility;
                count++;
            }
        }
        if (count < 3) {
            if (shouldLog) console.log("[Mocap Debug] Count of visible key landmarks too low:", count);
            return false;
        }
        const avgVis = totalVis / count;
        const shoulderDist = Math.abs(lm[11].x - lm[12].x);
        
        const ok = avgVis >= 0.75 && shoulderDist >= 0.10 && (lm[11].visibility >= 0.70 && lm[12].visibility >= 0.70);
        if (!ok && shouldLog) {
            console.log(`[Mocap Debug] Human fail details: avgVis=${avgVis.toFixed(2)} (req>=0.75), shoulderDist=${shoulderDist.toFixed(3)} (req>=0.10), leftShVis=${lm[11].visibility?.toFixed(2)} (req>=0.70), rightShVis=${lm[12].visibility?.toFixed(2)} (req>=0.70)`);
        }
        return ok;
    }

    /**
     * Callback when MediaPipe Pose completes image analysis on a frame
     */
    onPoseResults(results) {
        if (!this.firstResultReceived) {
            console.log("[Mocap Engine] ✅ First result RECEIVED from MediaPipe Pose!");
            this.firstResultReceived = true;
        }

        const landmarks = results.poseLandmarks || results.poseWorldLandmarks;

        // Run interactive state tracking (like spin detection) as long as we have raw landmarks,
        // even if visibility confidence is temporarily low (e.g. during a fast spin where back is turned)
        if (landmarks) {
            this.updateInteractiveStates(landmarks);
        }

        const humanPresent = this.isHumanPresent(results);

        if (!humanPresent) {
            if (!this.lastNoLandmarkLog || Date.now() - this.lastNoLandmarkLog > 2000) {
                console.log("[Mocap Engine] No human detected in frame (landmarks missing or low visibility). Waiting for person to enter frame...");
                this.lastNoLandmarkLog = Date.now();
            }
            if (typeof window.updateTrackingBadge === 'function') {
                window.updateTrackingBadge("warning", "未识别到人物姿态");
            }
            if (this.onPoseScore) this.onPoseScore(0); // reset score
            this.matchStartTime = null; // reset timer
            
            // Handle user absence (leaving frame or low landmark confidence)
            const now = Date.now();
            if (!this.absenceStartTime) {
                this.absenceStartTime = now;
            } else if (now - this.absenceStartTime > 5000) { // 5.0 seconds absence
                if (!this.isAbsent) {
                    this.isAbsent = true;
                    console.log("[Mocap Engine] User is ABSENT (no human detected)");
                    if (typeof window.onUserStill === 'function') {
                        window.onUserStill(true); // User is absent
                    }
                }
            }
            return;
        }
        
        // Reset absence on active detection
        if (this.isAbsent) {
            console.log("[Mocap Engine] User has RETURNED to the scene");
            this.isAbsent = false;
            if (typeof window.onUserActive === 'function') {
                window.onUserActive();
            }
        }
        
        if (!this.hasStarted) {
            console.log("[Mocap Engine] First time detection!");
            this.hasStarted = true;
            if (typeof window.onUserActive === 'function') {
                window.onUserActive();
            }
        }
        
        // Always fire detected callback if user is present
        if (typeof window.onUserDetected === 'function') {
            window.onUserDetected();
        }
        
        this.absenceStartTime = null;
        
        if (this.isPoseTriggerMode) {
            this.evaluatePoseMatching(landmarks);
        } else {
            if (typeof window.updateTrackingBadge === 'function') {
                window.updateTrackingBadge("success", "正在实时动捕中...");
            }
        }
    }
    
    /**
     * The Mathematical Retargeting Core:
     * Calculates vectors between joints and applies rotation Quaternions to Three.js bones
     */
    retargetSkeletalBones(lm) {
        // MediaPipe landmark index mapping helper
        const getVec = (index) => {
            const p = lm[index];
            // Invert Y to match Three.js coordinate system (Y goes up, MediaPipe Y goes down)
            // Invert Z to match coordinate depth
            return new THREE.Vector3(p.x, -p.y, -p.z);
        };
        
        try {
            // === 1. LEFT ARM ROTATION ===
            // Shoulder Pivot (Left Upper Arm)
            const shoulderL = getVec(11);
            const elbowL = getVec(13);
            const armDirL = new THREE.Vector3().subVectors(elbowL, shoulderL).normalize();
            // Default bone direction for left arm points horizontally outward: (1, 0, 0)
            const defaultUpperArmL = new THREE.Vector3(1, 0, 0);
            const qShoulderL = new THREE.Quaternion().setFromUnitVectors(defaultUpperArmL, armDirL);
            this.rotateJoint("shoulderL", qShoulderL);
            
            // Elbow Pivot (Left Forearm)
            const wristL = getVec(15);
            const forearmDirL = new THREE.Vector3().subVectors(wristL, elbowL).normalize();
            // Forearm rotates relative to upper arm, so we calculate rotation based on local bone projection
            const qElbowL = new THREE.Quaternion().setFromUnitVectors(armDirL, forearmDirL);
            this.rotateJoint("elbowL", qElbowL);
            
            // === 2. RIGHT ARM ROTATION ===
            // Shoulder Pivot (Right Upper Arm)
            const shoulderR = getVec(12);
            const elbowR = getVec(14);
            const armDirR = new THREE.Vector3().subVectors(elbowR, shoulderR).normalize();
            // Default bone direction for right arm points horizontally outward: (-1, 0, 0)
            const defaultUpperArmR = new THREE.Vector3(-1, 0, 0);
            const qShoulderR = new THREE.Quaternion().setFromUnitVectors(defaultUpperArmR, armDirR);
            this.rotateJoint("shoulderR", qShoulderR);
            
            // Elbow Pivot (Right Forearm)
            const wristR = getVec(16);
            const forearmDirR = new THREE.Vector3().subVectors(wristR, elbowR).normalize();
            const qElbowR = new THREE.Quaternion().setFromUnitVectors(armDirR, forearmDirR);
            this.rotateJoint("elbowR", qElbowR);
            
            // === 3. LEFT LEG ROTATION ===
            // Hip Pivot (Left Thigh)
            const hipL = getVec(23);
            const kneeL = getVec(25);
            const thighDirL = new THREE.Vector3().subVectors(kneeL, hipL).normalize();
            // Default bone direction for leg points straight down: (0, -1, 0)
            const defaultThighL = new THREE.Vector3(0, -1, 0);
            const qHipL = new THREE.Quaternion().setFromUnitVectors(defaultThighL, thighDirL);
            this.rotateJoint("hipL", qHipL);
            
            // Knee Pivot (Left Calf)
            const ankleL = getVec(27);
            const calfDirL = new THREE.Vector3().subVectors(ankleL, kneeL).normalize();
            const qKneeL = new THREE.Quaternion().setFromUnitVectors(thighDirL, calfDirL);
            this.rotateJoint("kneeL", qKneeL);
            
            // === 4. RIGHT LEG ROTATION ===
            // Hip Pivot (Right Thigh)
            const hipR = getVec(24);
            const kneeR = getVec(26);
            const thighDirR = new THREE.Vector3().subVectors(kneeR, hipR).normalize();
            const defaultThighR = new THREE.Vector3(0, -1, 0);
            const qHipR = new THREE.Quaternion().setFromUnitVectors(defaultThighR, thighDirR);
            this.rotateJoint("hipR", qHipR);
            
            // Knee Pivot (Right Calf)
            const ankleR = getVec(28);
            const calfDirR = new THREE.Vector3().subVectors(ankleR, kneeR).normalize();
            const qKneeR = new THREE.Quaternion().setFromUnitVectors(thighDirR, calfDirR);
            this.rotateJoint("kneeR", qKneeR);
            
            // === 5. SPINE/NECK (Simple Body Leaning) ===
            const shoulderMid = new THREE.Vector3().addVectors(shoulderL, shoulderR).multiplyScalar(0.5);
            const hipMid = new THREE.Vector3().addVectors(hipL, hipR).multiplyScalar(0.5);
            const spineDir = new THREE.Vector3().subVectors(shoulderMid, hipMid).normalize();
            const defaultSpine = new THREE.Vector3(0, 1, 0); // Upward
            const qSpine = new THREE.Quaternion().setFromUnitVectors(defaultSpine, spineDir);
            this.rotateJoint("spine", qSpine);
            
        } catch (e) {
            console.error("Error executing skeletal retargeting math:", e);
            // Do not re-throw here to prevent killing the MediaPipe frame loop!
        }
    }
    
    /**
     * Helper to rotate a specific joint node in Three.js using a Quaternion
     */
    rotateJoint(jointName, quaternion) {
        if (!this.avatar || !this.avatar.joints) return;
        const boneGroup = this.avatar.joints[jointName];
        if (boneGroup) {
            // Interpolate smoothly (Slerp) to reduce jitter from tracking noise
            boneGroup.quaternion.slerp(quaternion, 0.3);
        }
    }

    /**
     * Helper to calculate the 3D angle between three joints
     */
    calculateAngle(joint, ptA, ptB) {
        const vA = new THREE.Vector3().subVectors(ptA, joint).normalize();
        const vB = new THREE.Vector3().subVectors(ptB, joint).normalize();
        const dot = vA.dot(vB);
        const clampedDot = Math.max(-1, Math.min(1, dot));
        return Math.acos(clampedDot) * (180 / Math.PI);
    }

    /**
     * Evaluates if the user's current landmarks match the active target pose
     */
    evaluatePoseMatching(lm) {
        if (!lm) return;

        const getVec = (index) => {
            const p = lm[index];
            if (!p) return null; // Defensive guard: returns null if landmark is missing
            return new THREE.Vector3(p.x, -p.y, -p.z);
        };

        try {
            // Get target pose config
            const target = this.targetPoses[this.activePoseKey];
            if (!target) return;

            // 1. Calculate ONLY the requested joint angles on-demand to prevent out-of-frame landmark crashes!
            const jointValues = {};

            // Helper to check if a set of landmarks are all present
            const hasLandmarks = (indices) => {
                return indices.every(idx => lm[idx] !== undefined && lm[idx] !== null);
            };

            // --- ULTRA-ROBUST SPREAD ARMS (spread_arms) ---
            if (this.activePoseKey === 'spread_arms') {
                let isSpread = false;
                
                let leftArmRaised = false;
                if (hasLandmarks([11, 15])) {
                    const shoulderY = (lm[11].y + (hasLandmarks([12]) ? lm[12].y : lm[11].y)) / 2;
                    let hipY = 0.80;
                    if (lm[23]) hipY = lm[23].y;
                    const midTorsoY = (shoulderY + hipY) / 2;
                    if (lm[15].y < midTorsoY) leftArmRaised = true;
                }
                
                let rightArmRaised = false;
                if (hasLandmarks([12, 16])) {
                    const shoulderY = ((hasLandmarks([11]) ? lm[11].y : lm[12].y) + lm[12].y) / 2;
                    let hipY = 0.80;
                    if (lm[24]) hipY = lm[24].y;
                    const midTorsoY = (shoulderY + hipY) / 2;
                    if (lm[16].y < midTorsoY) rightArmRaised = true;
                }
                
                if (leftArmRaised || rightArmRaised) {
                    isSpread = true;
                }
                
                if (isSpread) {
                    this.isCurrentlySpreadingArms = true;
                    if (!this.matchStartTime) {
                        this.matchStartTime = Date.now();
                        if (typeof window.updateTrackingBadge === 'function') window.updateTrackingBadge("info", `抬起手臂蓄力中... 请保持`);
                    } else {
                        const duration = Date.now() - this.matchStartTime;
                        const progress = Math.min(100, Math.round((duration / 700) * 100)); // 700ms hold
                        if (typeof window.updateTrackingBadge === 'function') window.updateTrackingBadge("info", `材质变换蓄力: ${progress}%`);
                        
                        if (duration >= 700) { 
                            this.matchStartTime = null;
                            this.isPoseTriggerMode = false;
                            if (this.onPoseSuccess) this.onPoseSuccess(this.activePoseKey);
                        }
                    }
                } else {
                    this.isCurrentlySpreadingArms = false;
                    if (this.matchStartTime) {
                        this.matchStartTime = null;
                        if (typeof window.updateTrackingBadge === 'function') window.updateTrackingBadge("warning", `请举起任意一只手并保持片刻`);
                    }
                }
                return; // Override standard angle scoring
            }

            // --- RAISE ONE HAND (optimal_1) ---
            if (this.activePoseKey === 'optimal_1') {
                let isRaised = false;
                
                const hasLeft = hasLandmarks([15, 11]);
                const hasRight = hasLandmarks([16, 12]);
                
                // 1. LEFT wrist (15) is raised clearly above LEFT shoulder (11)
                const leftArmRaised = hasLeft && (lm[15].y < lm[11].y);
                // 2. RIGHT wrist (16) is raised clearly above RIGHT shoulder (12)
                const rightArmRaised = hasRight && (lm[16].y < lm[12].y);
                
                if (!this.poseLogCounter) this.poseLogCounter = 0;
                this.poseLogCounter++;
                const shouldLogPose = (this.poseLogCounter % 30 === 0);
                
                if (shouldLogPose) {
                    console.log(`[Mocap Debug] Pose optimal_1 match checking (Raise either hand):`);
                    console.log(`  LeftArm: hasLeft=${hasLeft}, LeftWristY=${hasLeft ? lm[15].y.toFixed(2) : 'N/A'}, LeftShoulderY=${hasLeft ? lm[11].y.toFixed(2) : 'N/A'} (Raised: ${leftArmRaised})`);
                    console.log(`  RightArm: hasRight=${hasRight}, RightWristY=${hasRight ? lm[16].y.toFixed(2) : 'N/A'}, RightShoulderY=${hasRight ? lm[12].y.toFixed(2) : 'N/A'} (Raised: ${rightArmRaised})`);
                }
                
                if (leftArmRaised || rightArmRaised) {
                    isRaised = true;
                }
                
                if (isRaised) {
                    if (!this.matchStartTime) {
                        this.matchStartTime = Date.now();
                        if (typeof window.updateTrackingBadge === 'function') window.updateTrackingBadge("info", `识别到举手！请保持...`);
                    } else {
                        const duration = Date.now() - this.matchStartTime;
                        const progress = Math.min(100, Math.round((duration / 800) * 100)); // 800ms hold
                        if (typeof window.updateTrackingBadge === 'function') window.updateTrackingBadge("info", `动作契合中: ${progress}%`);
                        
                        if (duration >= 800) { 
                            this.matchStartTime = null;
                            this.isPoseTriggerMode = false;
                            if (this.onPoseSuccess) this.onPoseSuccess(this.activePoseKey);
                        }
                    }
                } else {
                    if (this.matchStartTime) {
                        this.matchStartTime = null;
                        if (typeof window.updateTrackingBadge === 'function') window.updateTrackingBadge("warning", `姿势中断，请举起一只手并保持片刻`);
                    }
                }
                return; // Override standard angle scoring
            }

            // --- DECOUPLED ON-DEMAND JOINT CALCULATIONS ---
            
            // 1. Left Arm (tracked only if needed, independent of right arm)
            const needsLeftArm = ['leftShoulder', 'leftElbow'].some(k => k in target.angles);
            if (needsLeftArm && hasLandmarks([11, 13, 15])) {
                const shoulderL = getVec(11);
                const elbowL = getVec(13);
                const wristL = getVec(15);
                if (shoulderL && elbowL && wristL) {
                    jointValues.leftElbow = this.calculateAngle(elbowL, shoulderL, wristL);
                    const upVec = new THREE.Vector3(0, 1, 0);
                    const dirL = new THREE.Vector3().subVectors(elbowL, shoulderL).normalize();
                    jointValues.leftShoulder = Math.acos(Math.max(-1, Math.min(1, dirL.dot(upVec)))) * (180 / Math.PI);
                }
            }

            // 2. Right Arm (tracked only if needed, independent of left arm)
            const needsRightArm = ['rightShoulder', 'rightElbow'].some(k => k in target.angles);
            if (needsRightArm && hasLandmarks([12, 14, 16])) {
                const shoulderR = getVec(12);
                const elbowR = getVec(14);
                const wristR = getVec(16);
                if (shoulderR && elbowR && wristR) {
                    jointValues.rightElbow = this.calculateAngle(elbowR, shoulderR, wristR);
                    const upVec = new THREE.Vector3(0, 1, 0);
                    const dirR = new THREE.Vector3().subVectors(elbowR, shoulderR).normalize();
                    jointValues.rightShoulder = Math.acos(Math.max(-1, Math.min(1, dirR.dot(upVec)))) * (180 / Math.PI);
                }
            }

            // 3. Left Leg (tracked only if needed, independent of right leg)
            const needsLeftLeg = ['leftHip', 'leftKnee'].some(k => k in target.angles);
            if (needsLeftLeg) {
                if (hasLandmarks([23, 25, 27])) {
                    const hipL = getVec(23);
                    const kneeL = getVec(25);
                    const ankleL = getVec(27);
                    if (hipL && kneeL && ankleL) {
                        jointValues.leftKnee = this.calculateAngle(kneeL, hipL, ankleL);
                        const downVec = new THREE.Vector3(0, -1, 0);
                        const thighDirL = new THREE.Vector3().subVectors(kneeL, hipL).normalize();
                        jointValues.leftHip = Math.acos(Math.max(-1, Math.min(1, thighDirL.dot(downVec)))) * (180 / Math.PI);
                    }
                } else {
                    if (typeof window.updateTrackingBadge === 'function') {
                        window.updateTrackingBadge("warning", `⚠️ 请后退，确保左腿完全入镜`);
                    }
                }
            }

            // 4. Right Leg (tracked only if needed, independent of left leg)
            const needsRightLeg = ['rightHip', 'rightKnee'].some(k => k in target.angles);
            if (needsRightLeg) {
                if (hasLandmarks([24, 26, 28])) {
                    const hipR = getVec(24);
                    const kneeR = getVec(26);
                    const ankleR = getVec(28);
                    if (hipR && kneeR && ankleR) {
                        jointValues.rightKnee = this.calculateAngle(kneeR, hipR, ankleR);
                        const downVec = new THREE.Vector3(0, -1, 0);
                        const thighDirR = new THREE.Vector3().subVectors(kneeR, hipR).normalize();
                        jointValues.rightHip = Math.acos(Math.max(-1, Math.min(1, thighDirR.dot(downVec)))) * (180 / Math.PI);
                    }
                } else {
                    if (typeof window.updateTrackingBadge === 'function') {
                        window.updateTrackingBadge("warning", `⚠️ 请后退，确保右腿完全入镜`);
                    }
                }
            }

            // 2. Score only the joints defined in the active target pose (Tolerance: relaxed to 42 degrees for organic feel)
            const maxTolerance = 42;
            const calculateJointMatch = (userVal, targetVal) => {
                const diff = Math.abs(userVal - targetVal);
                return Math.max(0, 100 - (diff / maxTolerance) * 100);
            };

            let scores = [];
            for (const jointKey in target.angles) {
                if (jointValues[jointKey] !== undefined) {
                    const userVal = jointValues[jointKey];
                    const targetVal = target.angles[jointKey];
                    const jointScore = calculateJointMatch(userVal, targetVal);
                    scores.push(jointScore);
                } else {
                    // Fallback to 0 if a required landmark is missing or out of frame
                    scores.push(0);
                }
            }

            // Total score is the average match rate across all specified parameters
            const totalScore = scores.length > 0 ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length) : 0;

            // Send score to the frontend callback
            if (this.onPoseScore) {
                this.onPoseScore(totalScore);
            }

            // 3. Track duration for how long they hold this pose (Trigger threshold relaxed to 60%)
            if (totalScore >= 60) {
                if (!this.matchStartTime) {
                    this.matchStartTime = Date.now();
                    if (typeof window.updateTrackingBadge === 'function') {
                        window.updateTrackingBadge("info", `姿势契合！保持住... (Hold pose...)`);
                    }
                } else {
                    const duration = Date.now() - this.matchStartTime;
                    const progress = Math.min(100, Math.round((duration / this.requiredHoldTime) * 100));
                    if (typeof window.updateTrackingBadge === 'function') {
                        window.updateTrackingBadge("info", `时空共振中: ${progress}%`);
                    }

                    if (duration >= this.requiredHoldTime) {
                        this.matchStartTime = null; // reset
                        this.isPoseTriggerMode = false; // Disable trigger mode temporarily so they can watch the dance!
                        
                        if (this.onPoseSuccess) {
                            this.onPoseSuccess(this.activePoseKey);
                        }
                    }
                }
            } else {
                if (this.matchStartTime) {
                    this.matchStartTime = null;
                    if (typeof window.updateTrackingBadge === 'function') {
                        window.updateTrackingBadge("warning", `姿势中断，请根据剪影重新调整`);
                    }
                }
            }

        } catch (e) {
            console.error("Error during pose matching evaluation:", e);
        }
    }

    /**
     * Stateful Interaction Engine:
     * Detects 360-degree spins via shoulder crossover,
     * and detects stillness via rolling average landmark displacement.
     */
    updateInteractiveStates(lm) {
        if (!lm) return;

        const now = Date.now();
        const pL = lm[11]; // Left Shoulder
        const pR = lm[12]; // Right Shoulder

        // ================= 1. 360-DEGREE SPIN DETECTION & CONTINUOUS SPREAD ARMS =================
        if (lm && lm[11] && lm[12] && lm[15] && lm[16]) {
            let isSpread = false;
            const shoulderDist = Math.abs(lm[11].x - lm[12].x);
            const wristDist = Math.abs(lm[15].x - lm[16].x);
            
            const shoulderY = (lm[11].y + lm[12].y) / 2;
            let hipY = 0.85;
            if (lm[23] && lm[24]) hipY = (lm[23].y + lm[24].y) / 2;
            
            const wristsLifted = (lm[15].y < hipY + 0.05) && (lm[16].y < hipY + 0.05) && 
                                 (lm[15].y > (shoulderY - 0.45)) && (lm[16].y > (shoulderY - 0.45));
            
            const minShoulderX = Math.min(lm[11].x, lm[12].x);
            const maxShoulderX = Math.max(lm[11].x, lm[12].x);
            const minWristX = Math.min(lm[15].x, lm[16].x);
            const maxWristX = Math.max(lm[15].x, lm[16].x);
            
            const spreadOutwards = (minWristX < minShoulderX - 0.10 * shoulderDist) && 
                                   (maxWristX > maxShoulderX + 0.10 * shoulderDist);
            
            const spanWide = wristDist >= (shoulderDist * 1.5);
            
            if (wristsLifted && spreadOutwards && spanWide) {
                isSpread = true;
            }
            
            this.isCurrentlySpreadingArms = isSpread;
            if (isSpread) {
                this.lastSpreadArmsTime = Date.now();
            }
        }

        if (pL && pR) {
            const currentWidth = Math.abs(pL.x - pR.x);
            
            // Update rolling maximum shoulder width (slow decay, fast growth adaptation)
            if (currentWidth > this.maxShoulderWidth) {
                this.maxShoulderWidth = currentWidth;
            } else {
                // Slowly decay to adapt if user moves further away from the camera
                this.maxShoulderWidth = this.maxShoulderWidth * 0.998 + currentWidth * 0.002;
            }

            const relativeWidth = currentWidth / this.maxShoulderWidth;
            const currentOrder = pL.x > pR.x; // True if Left is to the right of Right (camera mirrored)

            if (this.baseOrder === null) {
                this.baseOrder = currentOrder; // Calibrate baseline on first frame
            }

            const isSwapped = currentOrder !== this.baseOrder;

            // Spin State Machine
            if (this.spinState === 0) {
                // State 0 -> 1: User starts turning sideways (shoulder width compresses)
                if (relativeWidth < 0.38) {
                    this.spinState = 1;
                    this.spinTimer = now;
                }
            } else {
                // Timeout Guard: If a full spin takes longer than 2.5 seconds, reset to State 0
                if (now - this.spinTimer > 2500) {
                    this.spinState = 0;
                    this.baseOrder = currentOrder; // Recalibrate order
                } else {
                    if (this.spinState === 1) {
                        // State 1 -> 2: User faces backward (shoulders widen, order is swapped)
                        if (relativeWidth > 0.58 && isSwapped) {
                            this.spinState = 2;
                        }
                    } else if (this.spinState === 2) {
                        // State 2 -> 3: User turns sideways again on the way back (width compresses)
                        if (relativeWidth < 0.38) {
                            this.spinState = 3;
                        }
                    } else if (this.spinState === 3) {
                        // State 3 -> 0 (Complete): User faces front again (width widens, order restored)
                        if (relativeWidth > 0.58 && !isSwapped) {
                            this.spinState = 0; // Reset
                            console.log(`[Mocap Engine] SPIN DETECTED! Duration: ${now - this.spinTimer}ms`);
                            
                            // Trigger global window callback
                            if (typeof window.onUserSpin === 'function') {
                                console.log('[USER TRIGGER 🟢] User triggered: SPIN');
                                window.onUserSpin();
                            }
                        }
                    }
                }
            }
        }

        // ================= 2. STILLNESS & ACTIVITY DETECTION =================
        let totalDisplacement = 0;
        const trackedIndices = [11, 12, 13, 14, 15, 16, 23, 24]; // Shoulders, elbows, wrists, hips
        let validPoints = 0;

        trackedIndices.forEach(idx => {
            const p = lm[idx];
            if (p) {
                const currentPos = new THREE.Vector3(p.x, p.y, p.z);
                if (this.prevLandmarks[idx]) {
                    totalDisplacement += currentPos.distanceTo(this.prevLandmarks[idx]);
                    validPoints++;
                }
                this.prevLandmarks[idx] = currentPos;
            }
        });

        if (validPoints > 0) {
            const avgDisplacement = totalDisplacement / validPoints;
            
            // Smooth the velocity using a rolling average filter
            this.movementIntensity = this.movementIntensity * 0.92 + avgDisplacement * 0.08;
            
            // Threshold for stillness (tuned to filter out camera pixel jitter)
            const STILL_THRESHOLD = 0.012; 
            
            if (this.movementIntensity < STILL_THRESHOLD) {
                if (!this.stillStartTime) {
                    this.stillStartTime = now;
                } else if (now - this.stillStartTime > 5000) { // Still for 5 seconds
                    if (!this.isStill) {
                        this.isStill = true;
                        console.log("[Mocap Engine] User is STILL for 5 seconds");
                        if (typeof window.onUserStill === 'function') {
                            window.onUserStill(false); // User is just still
                        }
                    }
                }
            } else {
                // User is active/moving
                this.stillStartTime = null;
                if (this.isStill) {
                    this.isStill = false;
                    console.log("[Mocap Engine] User is ACTIVE");
                    if (typeof window.onUserActive === 'function') {
                        window.onUserActive();
                    }
                }
            }
        }
    }

    async restartCamera() {
        if (this.camera) {
            console.log("[Mocap Engine] Forcefully restarting webcam camera helper...");
            try {
                this.camera.stop();
                await new Promise(resolve => setTimeout(resolve, 80)); // Short pause for hardware release
                await this.camera.start();
                console.log("[Mocap Engine] Camera helper restarted successfully.");
            } catch (err) {
                console.error("[Mocap Engine] Failed to restart camera helper:", err);
            }
        }
    }
}

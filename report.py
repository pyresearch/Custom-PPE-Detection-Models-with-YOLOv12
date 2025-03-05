import typer
import cv2
import supervision as sv
from ultralytics import YOLO
import pyresearch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime, timedelta
import numpy as np

# Load the model
model = YOLO("last.pt")
app = typer.Typer()

# Define classes
CLASSES = [
    "boots", "gloves", "goggles", "helmet", "no-helm", 
    "no_glove", "no_goggles", "no_helmet", "no_shoes", "person", "vest"
]

class DetectionTracker:
    def __init__(self):
        self.total_detections = 0
        self.class_counts = {cls: 0 for cls in CLASSES}
        self.processing_time = 0
        self.frame_count = 0
        self.person_safety_violations = []
        self.seen_persons = []  # Track unique persons by bounding box

    def is_new_person(self, person_box, threshold=50):
        """Check if this person is new by comparing bounding boxes"""
        for seen_box in self.seen_persons:
            if np.allclose(person_box, seen_box, atol=threshold):
                return False
        self.seen_persons.append(person_box)
        return True

def generate_pdf_report(tracker, output_file, pdf_file="worker_safety_kpi_report.pdf"):
    """Generate a KPI dashboard-style PDF report"""
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Worker Safety KPI Dashboard", styles['Heading1']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Paragraph(f"Output Video: {output_file}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Calculate KPIs
    total_incidents = len(tracker.person_safety_violations)
    
    # Critical Incidents (e.g., missing helmet or vest)
    critical_incidents = sum(1 for v in tracker.person_safety_violations if "helmet" in v["details"].lower() or "vest" in v["details"].lower())
    
    # Days Since Last Incident (placeholder)
    days_since_last_incident = 0 if total_incidents > 0 else 174
    
    # Incident Cost (simple formula: $500 per violation)
    incident_cost = total_incidents * 500

    # Severity Breakdown
    severity_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for violation in tracker.person_safety_violations:
        details = violation["details"].lower()
        if "goggles" in details or "gloves" in details:
            severity_counts["Low"] += 1
        elif "boots" in details:
            severity_counts["Medium"] += 1
        elif "vest" in details:
            severity_counts["High"] += 1
        elif "helmet" in details:
            severity_counts["Critical"] += 1

    # Type of Incident
    type_counts = {
        "No Helmet": 0, "No Goggles": 0, "No Gloves": 0, 
        "No Boots": 0, "No Vest": 0
    }
    for violation in tracker.person_safety_violations:
        details = violation["details"]
        for violation_type in type_counts.keys():
            if violation_type in details:
                type_counts[violation_type] += 1

    # Injury Consequence
    consequence_counts = {
        "Lost Time": 0, "Medical Case": 0, "First Aid": 0, 
        "No Treatment": 0, "Lost Days": 0
    }
    for violation in tracker.person_safety_violations:
        details = violation["details"].lower()
        if "helmet" in details:
            consequence_counts["Lost Time"] += 1
            consequence_counts["Lost Days"] += 1
        elif "vest" in details:
            consequence_counts["Medical Case"] += 1
        elif "boots" in details:
            consequence_counts["First Aid"] += 1
        elif "goggles" in details or "gloves" in details:
            consequence_counts["No Treatment"] += 1

    # KPI Summary
    elements.append(Paragraph("KPI Summary", styles['Heading2']))
    kpi_data = [
        ["# Incidents", "# Critical Incidents", "Days Since Last Incident", "Incident Cost"],
        [str(total_incidents), str(critical_incidents), str(days_since_last_incident), f"${incident_cost:,}"]
    ]
    kpi_table = Table(kpi_data)
    kpi_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
    ])
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))

    # Severity Breakdown
    elements.append(Paragraph("Severity", styles['Heading2']))
    severity_data = [["Severity", "Count"]]
    for severity, count in severity_counts.items():
        severity_data.append([severity, str(count)])
    
    severity_table = Table(severity_data)
    severity_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
    ])
    elements.append(severity_table)
    elements.append(Spacer(1, 12))

    # Type of Incident
    elements.append(Paragraph("Type of Incident", styles['Heading2']))
    type_data = [["Type", "Count"]]
    for incident_type, count in type_counts.items():
        type_data.append([incident_type, str(count)])
    
    type_table = Table(type_data)
    type_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
    ])
    elements.append(type_table)
    elements.append(Spacer(1, 12))

    # Injury Consequence
    elements.append(Paragraph("Injury Consequence", styles['Heading2']))
    consequence_data = [["Consequence", "Count"]]
    for consequence, count in consequence_counts.items():
        consequence_data.append([consequence, str(count)])
    
    consequence_table = Table(consequence_data)
    consequence_table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
    ])
    elements.append(consequence_table)
    elements.append(Spacer(1, 12))

    # Safety Violations Timeline
    elements.append(Paragraph("Safety Violations Timeline", styles['Heading2']))
    if tracker.person_safety_violations:
        violation_data = [["Timestamp", "Event", "Details"]]
        for event in tracker.person_safety_violations:
            violation_data.append([
                event["timestamp"],
                event["event"],
                event["details"]
            ])
        
        violation_table = Table(violation_data)
        violation_table.setStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ])
        elements.append(violation_table)
    else:
        elements.append(Paragraph("No safety violations detected.", styles['Normal']))

    doc.build(elements)
    print(f"PDF report generated: {pdf_file}")

def process_webcam(output_file="output.mp4", generate_pdf=True):
    cap = cv2.VideoCapture("demo.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    tracker = DetectionTracker()

    # Get video duration for timestamp calculation
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_seconds = total_frames / fps if fps > 0 else 0
    start_time = datetime.now()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = cv2.getTickCount()
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        elapsed_seconds = current_frame / fps if fps > 0 else 0
        current_time = (start_time + timedelta(seconds=elapsed_seconds)).strftime("%H:%M:%S.%f")[:-3]
        
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        tracker.frame_count += 1
        tracker.total_detections += len(detections)

        # Debug: Print raw detection results
        if len(detections) > 0:
            print(f"Frame {current_frame}: {len(detections)} detections")

        # Process detections
        current_frame_detections = {cls: [] for cls in CLASSES}
        if hasattr(detections, 'class_id') and hasattr(detections, 'xyxy'):
            for i, class_id in enumerate(detections.class_id):
                class_name = model.names[class_id] if class_id < len(model.names) else f"unknown_{class_id}"
                if class_name in CLASSES:
                    tracker.class_counts[class_name] += 1
                    current_frame_detections[class_name].append(detections.xyxy[i])
                else:
                    print(f"Warning: Detected class {class_name} not in defined CLASSES")

            # Check safety compliance for persons (only for new persons)
            if "person" in current_frame_detections and len(current_frame_detections["person"]) > 0:
                for person_box in current_frame_detections["person"]:
                    if tracker.is_new_person(person_box):
                        # Track violations for this person
                        person_violations = set()

                        # Check for helmet
                        has_helmet = any(
                            (person_box[0] <= h_box[2] and person_box[2] >= h_box[0] and
                             person_box[1] <= h_box[3] and person_box[3] >= h_box[1])
                            for h_box in current_frame_detections.get("helmet", [])
                        )
                        if not has_helmet and "No Helmet" not in person_violations:
                            person_violations.add("No Helmet")
                            tracker.person_safety_violations.append({
                                "timestamp": current_time,
                                "event": "Safety Violation",
                                "details": "No Helmet"
                            })
                    
                        # Check for goggles
                        has_goggles = any(
                            (person_box[0] <= g_box[2] and person_box[2] >= g_box[0] and
                             person_box[1] <= g_box[3] and person_box[3] >= g_box[1])
                            for g_box in current_frame_detections.get("goggles", [])
                        )
                        if not has_goggles and "No Goggles" not in person_violations:
                            person_violations.add("No Goggles")
                            tracker.person_safety_violations.append({
                                "timestamp": current_time,
                                "event": "Safety Violation",
                                "details": "No Goggles"
                            })
                    
                        # Check for boots
                        has_boots = any(
                            (person_box[0] <= b_box[2] and person_box[2] >= b_box[0] and
                             person_box[1] <= b_box[3] and person_box[3] >= b_box[1])
                            for b_box in current_frame_detections.get("boots", [])
                        )
                        if not has_boots and "No Boots" not in person_violations:
                            person_violations.add("No Boots")
                            tracker.person_safety_violations.append({
                                "timestamp": current_time,
                                "event": "Safety Violation",
                                "details": "No Boots"
                            })

                        # Check for gloves
                        has_gloves = any(
                            (person_box[0] <= g_box[2] and person_box[2] >= g_box[0] and
                             person_box[1] <= g_box[3] and person_box[3] >= g_box[1])
                            for g_box in current_frame_detections.get("gloves", [])
                        )
                        if not has_gloves and "No Gloves" not in person_violations:
                            person_violations.add("No Gloves")
                            tracker.person_safety_violations.append({
                                "timestamp": current_time,
                                "event": "Safety Violation",
                                "details": "No Gloves"
                            })

                        # Check for vest
                        has_vest = any(
                            (person_box[0] <= v_box[2] and person_box[2] >= v_box[0] and
                             person_box[1] <= v_box[3] and person_box[3] >= v_box[1])
                            for v_box in current_frame_detections.get("vest", [])
                        )
                        if not has_vest and "No Vest" not in person_violations:
                            person_violations.add("No Vest")
                            tracker.person_safety_violations.append({
                                "timestamp": current_time,
                                "event": "Safety Violation",
                                "details": "No Vest"
                            })

            # Check for explicit "no_xxx" detections in each frame
            for class_name, boxes in current_frame_detections.items():
                if class_name.startswith("no_") and len(boxes) > 0:
                    violation_type = f"No {class_name.replace('no_', '').capitalize()}"
                    tracker.person_safety_violations.append({
                        "timestamp": current_time,
                        "event": "Safety Violation",
                        "details": violation_type
                    })

        # Annotate and process frame
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        frame_end = cv2.getTickCount()
        tracker.processing_time += ((frame_end - frame_start) / cv2.getTickFrequency()) * 1000

        out.write(annotated_frame)
        cv2.imshow("Webcam", annotated_frame)
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if generate_pdf:
        generate_pdf_report(tracker, output_file)

@app.command()
def webcam(output_file: str = "output.mp4", pdf_report: bool = True):
    typer.echo("Starting webcam processing...")
    process_webcam(output_file, pdf_report)

if __name__ == "__main__":
    app()
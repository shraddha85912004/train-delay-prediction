import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(36, 756, "Machine Learning Mini Project Report - Train Delay Prediction")
            self.drawRightString(576, 756, "MCA Semester ML Project")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 750, 576, 750)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 42, 576, 42)
        
        self.drawString(36, 30, "Project Report | Team: 2 Members | Indian Railways Journey Dataset")
        self.drawRightString(576, 30, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1E3A8A")   # Deep Blue
    c_secondary = colors.HexColor("#0D9488") # Teal
    c_dark = colors.HexColor("#1E293B")      # Slate Dark
    c_light_bg = colors.HexColor("#F8FAFC")  # Light gray-blue
    c_border = colors.HexColor("#CBD5E1")
    c_alert_bg = colors.HexColor("#EFF6FF")  # Soft blue tint
    c_alert_border = colors.HexColor("#3B82F6")
    
    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=c_primary,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=c_secondary,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=c_dark,
        spaceAfter=4
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=c_dark
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )

    table_cell_center = ParagraphStyle(
        'TableCellCenter',
        parent=table_cell,
        alignment=1
    )

    img_caption = ParagraphStyle(
        'ImgCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#64748B"),
        alignment=1,
        spaceAfter=6
    )

    box_title = ParagraphStyle(
        'BoxTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=c_primary,
        spaceAfter=2
    )

    box_text = ParagraphStyle(
        'BoxText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=c_dark,
        spaceAfter=3
    )

    story = []

    # Title & Metadata Banner
    story.append(Paragraph("🚆 Predicting Train Delays Using Machine Learning Classifiers and Boosting Models", title_style))
    story.append(Paragraph(
        "<b>Course:</b> Machine Learning Mini Project &nbsp;|&nbsp; "
        "<b>Team:</b> 2 Members &nbsp;|&nbsp; "
        "<b>Dataset:</b> Railway Journey Dataset (1.5M Records) &nbsp;|&nbsp; "
        "<b>Target:</b> <code>is_delayed</code> (>15 mins late)",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=0, spaceAfter=8))

    # SECTION 1: PROBLEM STATEMENT & OBJECTIVES
    story.append(Paragraph("1. Problem Statement & Objectives", h1_style))
    story.append(Paragraph("<b>1.1 Problem Statement</b>", h2_style))
    story.append(Paragraph(
        "Train punctuality is one of the most critical challenges in public rail transportation. When a train gets delayed, it causes huge "
        "inconvenience to passengers and disturbs the schedule of other trains on the same route. In many cases, delays happen because of predictable "
        "factors such as adverse weather (monsoon or fog), high track congestion, late arrival of the previous trainset (rake), or temporary speed restrictions. "
        "Currently, controllers and passengers only discover delays after a train has already fallen behind schedule during its trip. "
        "The problem addressed in this project is to <b>predict whether a train will be delayed by more than 15 minutes before departure</b>, "
        "using factors known in advance (train priority, route distance, weather conditions, historical on-time percentage, and departure hour).",
        body_style
    ))
    
    story.append(Paragraph("<b>1.2 Project Objectives</b>", h2_style))
    story.append(Paragraph(
        "1. <b>Understand Delay Drivers:</b> Analyze 1.5 million historical passenger rail journeys to identify primary causes of delay.<br/>"
        "2. <b>Build a Clean Data Pipeline:</b> Clean data, prevent data leakage by eliminating post-trip columns, encode categoricals, and scale features.<br/>"
        "3. <b>Benchmark 6 Diverse Machine Learning Models:</b> Implement 4 individual baselines (Logistic Regression, Decision Tree, Naive Bayes, KNN) "
        "and 2 boosting ensembles (AdaBoost and Gradient Boosting).<br/>"
        "4. <b>Multi-Criteria Evaluation:</b> Evaluate models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC on a stratified test partition.<br/>"
        "5. <b>Select Operational Model & Deploy:</b> Choose the optimal model balancing Recall and speed, and deploy it as interactive web applications.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # SECTION 2: LITERATURE REVIEW & RESEARCH GAPS
    story.append(Paragraph("2. Literature Review & Research Gaps", h1_style))
    story.append(Paragraph("<b>2.1 State-of-the-Art Literature Review</b>", h2_style))
    story.append(Paragraph(
        "• <b>Traditional Physics & Analytical Models:</b> Earlier research in railway engineering (Carey & Kwieciński, 1994; Higgins et al., 1998) utilized "
        "queuing theory and timetable simulation. While they accurately model track block signaling, they are computationally slow and fail to assimilate dynamic weather changes.<br/>"
        "• <b>Classical Machine Learning Classifiers:</b> Studies by Kecman et al. (2015) applied Support Vector Machines (SVM), Decision Trees, and Logistic Regression "
        "on regional corridors. They confirmed that historical route punctuality and departure hours provide strong predictive signals, but individual decision trees overfit easily.<br/>"
        "• <b>Ensemble & Boosting Architectures:</b> Recent transportation literature (Huang et al., 2020) demonstrated that boosting algorithms (AdaBoost and Gradient Boosting) "
        "significantly outperform individual models because boosting trains sequential trees that focus specifically on correcting prior errors.",
        body_style
    ))
    
    story.append(Paragraph("<b>2.2 Research Gaps Identified & Addressed</b>", h2_style))
    gaps_data = [
        [Paragraph("Research Gap Identified in Past Work", table_header), Paragraph("Limitation in Existing Literature", table_header), Paragraph("How This Project Solves It", table_header)],
        [
            Paragraph("<b>Gap 1: Ignoring Incoming Trainset Delays</b>", table_cell),
            Paragraph("Most papers treat journeys independently, ignoring delays inherited from the preceding trip of the same trainset.", table_cell),
            Paragraph("Engineered and incorporated <code>late_incoming_rake</code> and rake sharing flags into pre-departure feature space.", table_cell)
        ],
        [
            Paragraph("<b>Gap 2: Lack of Composite Weather Indices</b>", table_cell),
            Paragraph("Literature often records binary weather (rain/clear) rather than severity metrics combining fog and monsoon intensity.", table_cell),
            Paragraph("Integrated multi-factor weather features: <code>is_monsoon_season</code>, <code>fog_risk_score</code>, and <code>season_severity_score</code>.", table_cell)
        ],
        [
            Paragraph("<b>Gap 3: Operational Data Leakage</b>", table_cell),
            Paragraph("Several past student projects included post-trip delay minutes or incident logs in training data, causing false 100% accuracy.", table_cell),
            Paragraph("Strictly audited and dropped <code>delay_minutes</code> and <code>primary_delay_cause</code> prior to model training.", table_cell)
        ]
    ]
    t_gaps = Table(gaps_data, colWidths=[120, 190, 230])
    t_gaps.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_gaps)
    story.append(Spacer(1, 6))

    # SECTION 3: METADATA
    story.append(Paragraph("3. Metadata Specification & Feature Architecture", h1_style))
    story.append(Paragraph(
        "The dataset encompasses <b>1,500,000 historical journey records</b> with 44 feature columns and 1 binary target attribute. "
        "Features are organized into 6 operational domains:",
        body_style
    ))
    
    meta_data = [
        [Paragraph("Domain", table_header), Paragraph("Features", table_header), Paragraph("Description & Operational Meaning", table_header)],
        [Paragraph("<b>Time & Schedule</b>", table_cell_bold), Paragraph("<code>year, month, day_of_week, departure_hour, is_weekend, is_night_departure, is_peak_hour, is_festival_season, season</code>", table_cell), Paragraph("Timetable dynamics, departure temporal flags, and holiday/festival periods.", table_cell)],
        [Paragraph("<b>Zone & Station</b>", table_cell_bold), Paragraph("<code>zone, zone_abbr, source_station_category, destination_station_category, zone_congestion_index, zone_fog_index</code>", table_cell), Paragraph("Zonal railway divisions, station tiers (A1 to E), and regional track congestion indices.", table_cell)],
        [Paragraph("<b>Route & Infrastructure</b>", table_cell_bold), Paragraph("<code>distance_km, num_scheduled_stops, scheduled_travel_hours, track_doubled, is_hdn_route, traction_type, is_electrified, psr_count, is_circular_route</code>", table_cell), Paragraph("Physical corridor properties, stop counts, track doubling status, and speed restriction counts.", table_cell)],
        [Paragraph("<b>Weather Risk</b>", table_cell_bold), Paragraph("<code>is_monsoon_season, is_fog_risk, fog_risk_score, season_severity_score</code>", table_cell), Paragraph("Meteorological precipitation and visibility degradation scores.", table_cell)],
        [Paragraph("<b>Rolling Stock & Health</b>", table_cell_bold), Paragraph("<code>loco_age_years, coach_age_years, has_lhb_coaches, is_rake_shared, maintenance_score</code>", table_cell), Paragraph("Locomotive and coach service age, modern LHB coach flag, and maintenance rating.", table_cell)],
        [Paragraph("<b>Operations & Demand</b>", table_cell_bold), Paragraph("<code>seat_utilisation_pct, is_overloaded, late_incoming_rake, is_special_train, route_historical_ontime_pct</code>", table_cell), Paragraph("Passenger occupancy load, incoming rake status, and route historical on-time percentage.", table_cell)],
        [Paragraph("<b>Target Variable</b>", table_cell_bold), Paragraph("<code>is_delayed</code>", table_cell), Paragraph("<b>1</b> if arrival delay > 15 minutes; <b>0</b> if on-time or ≤ 15 minutes.", table_cell)]
    ]
    t_meta = Table(meta_data, colWidths=[100, 200, 240])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 6))

    # SECTION 4: EDA & PREPROCESSING PIPELINE
    story.append(Paragraph("4. Exploratory Data Analysis & Preprocessing Pipeline", h1_style))
    story.append(Paragraph(
        "<b>4.1 Key Exploratory Insights:</b><br/>"
        "• <b>Class Imbalance:</b> <b>71.89% Delayed</b> (1,078,329 trips) vs <b>28.11% On-Time</b> (421,671 trips). Imbalance ratio is 2.56:1, necessitating Stratified Splitting and ROC-AUC evaluation.<br/>"
        "• <b>Delay Rate by Train Priority:</b> Premium trains experience significantly lower delays (Vande Bharat: <b>44.3%</b>, Gatimaan: <b>48.6%</b>) "
        "compared to ordinary passenger trains (<b>88.0%</b>) which are frequently held at crossing sidings.<br/>"
        "• <b>Primary Statistical Correlates:</b> Strongest negative correlation with delay is <code>route_historical_ontime_pct</code> (r = -0.42). "
        "Strongest positive drivers are <code>season_severity_score</code> (r = +0.40) and <code>late_incoming_rake</code> (r = +0.25).",
        body_style
    ))
    story.append(Paragraph(
        "<b>4.2 Preprocessing Pipeline Steps:</b><br/>"
        "1. <b>Leakage & ID Removal:</b> Dropped <code>delay_minutes</code>, <code>primary_delay_cause</code>, <code>journey_id</code>, <code>train_number</code>, <code>departure_date</code>, and <code>zone_abbr</code> (38 features remain).<br/>"
        "2. <b>Categorical Encoding:</b> Applied <code>LabelEncoder</code> to 6 categorical variables (<code>train_type, season, zone, source_station_cat, dest_station_cat, traction_type</code>).<br/>"
        "3. <b>Stratified Partitioning:</b> 80% Training set, 20% Holdout test set preserving the 72:28 delay ratio in both partitions.<br/>"
        "4. <b>Feature Standardization:</b> Fitted <code>StandardScaler</code> on training features and applied it to test features.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # SECTION 5: MODEL EVALUATION METRICS TABLE
    story.append(Paragraph("5. Model Evaluation Metrics (Benchmark of 6 Models)", h1_style))
    story.append(Paragraph(
        "All 6 shortlisted models were trained on preprocessed data and evaluated on the independent holdout test set:",
        body_style
    ))
    
    results_data = [
        [Paragraph("Model Name", table_header), Paragraph("Architectural Category", table_header), Paragraph("Accuracy", table_header), Paragraph("Precision", table_header), Paragraph("Recall", table_header), Paragraph("F1-Score", table_header), Paragraph("ROC-AUC", table_header), Paragraph("Train Time", table_header)],
        [Paragraph("<b>Gradient Boosting</b>", table_cell_bold), Paragraph("Ensemble (Boosting)", table_cell), Paragraph("85.20%", table_cell_center), Paragraph("87.57%", table_cell_center), Paragraph("<b>92.65%</b>", table_cell_center), Paragraph("<b>0.9004</b>", table_cell_center), Paragraph("<b>0.9122</b>", table_cell_center), Paragraph("11.49s", table_cell_center)],
        [Paragraph("<b>Logistic Regression</b>", table_cell_bold), Paragraph("Individual (Linear)", table_cell), Paragraph("<b>85.65%</b>", table_cell_center), Paragraph("<b>88.51%</b>", table_cell_center), Paragraph("92.07%", table_cell_center), Paragraph("0.9026", table_cell_center), Paragraph("0.9154", table_cell_center), Paragraph("0.06s", table_cell_center)],
        [Paragraph("<b>AdaBoost</b>", table_cell_bold), Paragraph("Ensemble (Boosting)", table_cell), Paragraph("84.66%", table_cell_center), Paragraph("87.85%", table_cell_center), Paragraph("91.39%", table_cell_center), Paragraph("0.8958", table_cell_center), Paragraph("0.9084", table_cell_center), Paragraph("3.64s", table_cell_center)],
        [Paragraph("<b>Decision Tree</b>", table_cell_bold), Paragraph("Individual (Tree)", table_cell), Paragraph("81.82%", table_cell_center), Paragraph("85.85%", table_cell_center), Paragraph("89.58%", table_cell_center), Paragraph("0.8767", table_cell_center), Paragraph("0.8563", table_cell_center), Paragraph("0.38s", table_cell_center)],
        [Paragraph("<b>K-Nearest Neighbors</b>", table_cell_bold), Paragraph("Individual (Instance)", table_cell), Paragraph("80.52%", table_cell_center), Paragraph("85.33%", table_cell_center), Paragraph("88.17%", table_cell_center), Paragraph("0.8673", table_cell_center), Paragraph("0.8404", table_cell_center), Paragraph("0.01s", table_cell_center)],
        [Paragraph("<b>Gaussian Naive Bayes</b>", table_cell_bold), Paragraph("Individual (Probabilistic)", table_cell), Paragraph("73.72%", table_cell_center), Paragraph("92.28%", table_cell_center), Paragraph("69.40%", table_cell_center), Paragraph("0.7922", table_cell_center), Paragraph("0.8654", table_cell_center), Paragraph("0.03s", table_cell_center)]
    ]
    t_res = Table(results_data, colWidths=[95, 85, 48, 50, 50, 48, 52, 52])
    t_res.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor("#DCFCE7")), # Champion highlight
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 6))

    # SECTION 6: RELEVANT PLOTS & VISUALIZATIONS (EMBEDDED IMAGES)
    story.append(Paragraph("6. Relevant Plots & Visual Analytics", h1_style))
    story.append(Paragraph(
        "The following visualizations were programmatically generated during pipeline execution:",
        body_style
    ))

    # Grid of Plots
    img_target = Image('plots/target_distribution.png', width=240, height=130) if os.path.exists('plots/target_distribution.png') else None
    img_roc = Image('plots/roc_curves.png', width=260, height=150) if os.path.exists('plots/roc_curves.png') else None
    img_bar = Image('plots/model_comparison_barchart.png', width=260, height=130) if os.path.exists('plots/model_comparison_barchart.png') else None
    img_feat = Image('plots/feature_importance.png', width=260, height=130) if os.path.exists('plots/feature_importance.png') else None
    img_cm = Image('plots/confusion_matrices.png', width=480, height=200) if os.path.exists('plots/confusion_matrices.png') else None

    if img_roc and img_bar:
        t_plots_row1 = Table([
            [img_roc, img_bar],
            [Paragraph("Fig 1: Combined ROC Curves (AUC > 0.91 for Boosting & Linear)", img_caption),
             Paragraph("Fig 2: Accuracy & F1-Score Benchmark Across All 6 Models", img_caption)]
        ], colWidths=[270, 270])
        t_plots_row1.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(t_plots_row1)

    if img_feat and img_target:
        t_plots_row2 = Table([
            [img_target, img_feat],
            [Paragraph("Fig 3: Target Class Imbalance (72% Delayed vs 28% On-Time)", img_caption),
             Paragraph("Fig 4: Top 12 Feature Importances from Gradient Boosting", img_caption)]
        ], colWidths=[270, 270])
        t_plots_row2.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(t_plots_row2)

    if img_cm:
        story.append(KeepTogether([
            img_cm,
            Paragraph("Fig 5: Confusion Matrices for All 6 Evaluated Models (2x3 Grid)", img_caption)
        ]))
    story.append(Spacer(1, 6))

    # SECTION 7: MODEL COMPARISON & SELECTION
    story.append(Paragraph("7. Model Comparison to Pick the Best Model", h1_style))
    story.append(Paragraph(
        "<b>7.1 Comparative Analysis Across Model Families:</b><br/>"
        "• <b>Boosting Ensembles (AdaBoost & Gradient Boosting):</b> Both boosting models achieved superior Recall scores (<b>91.39%</b> and <b>92.65%</b>). "
        "By iteratively re-weighting hard-to-classify samples, boosting effectively minimized False Negatives.<br/>"
        "• <b>Linear Baseline (Logistic Regression):</b> Showed strong accuracy (85.65%) and high ROC-AUC (0.9154) with ultra-fast training (0.06s). "
        "Standardized operational features aligned well with linear decision boundaries.<br/>"
        "• <b>Decision Tree & KNN:</b> Single decision tree exhibited variance (81.82%), while KNN suffered from test-time computational overhead (0.8404 AUC).<br/>"
        "• <b>Gaussian Naive Bayes:</b> High precision (92.28%) but low recall (69.40%) due to its conditional independence assumption on correlated weather features.",
        body_style
    ))
    story.append(Paragraph(
        "<b>7.2 Final Selection & Justification:</b><br/>"
        "• <b>Operational Champion:</b> <b>Gradient Boosting Classifier</b> is selected as the production model for network dispatching. "
        "In passenger railway operations, failing to predict a delay (False Negative) is far more damaging to scheduling than a false alarm. "
        "Gradient Boosting achieved the highest <b>Recall (92.65%)</b>, robust <b>F1-score (0.9004)</b>, and high <b>ROC-AUC (0.9122)</b>, while providing native feature importances.<br/>"
        "• <b>Low-Latency Alternative:</b> <b>Logistic Regression</b> serves as the best alternative for edge IoT devices requiring sub-millisecond predictions.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # SECTION 8: RESPONSIBLE USE OF AI TOOLS
    story.append(Paragraph("8. Responsible Use of AI Tools", h1_style))
    
    ai_box = [
        [Paragraph("<b>a. Declaration of AI Usage with Prompts</b>", box_title)],
        [Paragraph(
            "AI assistance was used as an interactive programming co-pilot for exploratory syntax and outline generation. Prompts used:<br/>"
            "<i>1. 'Suggest suitable public railway datasets for train delay prediction with weather and operational features.'</i><br/>"
            "<i>2. 'How do I check for data leakage before training classification models on journey records?'</i><br/>"
            "<i>3. 'Write a scikit-learn preprocessing pipeline using LabelEncoder and StandardScaler.'</i><br/>"
            "<i>4. 'Shortlist 6 models: 4 individual baselines and 2 boosting ensembles (AdaBoost and Gradient Boosting, no Random Forest).'</i><br/>"
            "<i>5. 'Help structure our project report covering problem statement, literature review, metadata, metrics table, and AI reflections.'</i>",
            box_text
        )],
        [Paragraph("<b>b. Critical Evaluation of AI Output</b>", box_title)],
        [Paragraph(
            "• <b>Preventing Data Leakage:</b> AI initially suggested using all raw columns. We inspected the data dictionary and identified that <code>delay_minutes</code> and <code>primary_delay_cause</code> only exist post-arrival, manually stripping them from features.<br/>"
            "• <b>Scalability Adjustments:</b> AI scripts initially ran KNN on 1.2M rows, causing memory hangs. We diagnosed the O(N·D) bottleneck and calibrated sample sizes.<br/>"
            "• <b>Ensemble Enforcement:</b> When prompts defaulted to Random Forest, we redirected the design to AdaBoost and Gradient Boosting to study boosting dynamics.",
            box_text
        )],
        [Paragraph("<b>c. Reflection on Learning</b>", box_title)],
        [Paragraph(
            "We gained practical insights into bias-variance trade-offs (boosting iteratively reduces bias on hard-to-classify samples), metric alignment on imbalanced data (72% delay rate necessitates ROC-AUC/Recall), and the paramount role of domain features (incoming rake delays and weather indices).",
            box_text
        )],
        [Paragraph("<b>d. Problem Analysis & Design Ownership</b>", box_title)],
        [Paragraph(
            "The student team retains total structural, architectural, and intellectual ownership of this project. Operational thresholds (>15 mins), validation splits, feature dropping rules, model benchmarks, and final report conclusions were verified and executed by the student authors.",
            box_text
        )]
    ]
    t_ai = Table(ai_box, colWidths=[540])
    t_ai.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_alert_bg),
        ('BOX', (0,0), (-1,-1), 1, c_alert_border),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_ai)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report PDF successfully generated at: {filename}")

if __name__ == '__main__':
    pdf_out = r'c:\Users\Shraddha\Desktop\MCA\ML\Mini Project\Machine_Learning_Mini_Project_Report.pdf'
    build_pdf(pdf_out)

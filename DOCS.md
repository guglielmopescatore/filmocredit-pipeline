# 📚 FilmoCredit User Manual

## 🎯 What is FilmoCredit?

FilmoCredit is an AI-powered tool that automatically extracts and validates credits from TV episodes and movies. It uses advanced computer vision and optical character recognition (OCR) to identify credit scenes, extract text, and cross-reference names with the IMDB database.

## 🚀 Getting Started

### 1. Launch the Application
After installation, start FilmoCredit by:
- **Windows**: Run `FilmoCredit.bat`
- **Linux/macOS**: Run `./FilmoCredit.sh`

The application will open in your web browser at `http://localhost:8501`

### 2. Add Your Videos
Place your video files (`.mp4`, `.mkv`, `.avi`, `.mov`) in the `data/raw/` folder within your FilmoCredit installation directory.

## 🎛️ Interface Overview

FilmoCredit has three main sections accessible via tabs:

### ⚙️ Setup & Run Pipeline
Configure and run the credit extraction process

### ✏️ Review & Edit Credits  
Review, edit, and fix extracted credits

### 📊 Logs
View processing logs and troubleshoot issues

---

## ⚙️ Setup & Configuration

### Language Settings
**OCR Language**: Choose the primary language of your video credits:
- **Italian** (`it`) - for Italian content
- **English** (`en`) - for English content  
- **Chinese** (`ch`) - for Chinese content

### Scene Detection Methods
Choose how FilmoCredit analyzes your videos:

#### 📋 By Scene Count (Recommended)
- **Start Scenes**: Number of scenes to analyze at the beginning (default: 100)
- **End Scenes**: Number of scenes to analyze at the end (default: 100)
- Best for: Most TV shows and movies where credits appear at start/end

#### ⏱️ By Time Duration
- **Start Minutes**: Time duration from beginning to analyze (default: 7.0 minutes)
- **End Minutes**: Time duration from end to analyze (default: 7.0 minutes)  
- Best for: When you know the approximate duration of credit sequences

#### 🎬 Whole Episode
- Analyzes the entire video from start to finish
- Best for: Short videos or when credits appear throughout

### Stopwords Configuration
Add words that appear as channel logos, watermarks, or overlays that should be ignored during credit extraction:
- Add one word per line in the sidebar text area
- Default words: "RAI", "BBC", "HBO"
- Click **Save Stopwords** to apply changes

### Video Preview
- Check **📺 Enable Video Preview** to see a preview of selected videos
- Select a video from the dropdown to preview
- Helps verify you're processing the correct content

---

## 🔄 Processing Pipeline

The credit extraction works in 4 sequential steps:

### Step 1: Identify Candidate Scenes
**What it does**: Uses AI to automatically detect scenes that likely contain credits
- Analyzes video based on your scene detection settings
- Identifies scenes with text overlays, scrolling credits, or static credit frames
- Creates a list of candidate scenes for further processing

**When to use**: Always run this first for any new video

### Step 2: Analyze Scene Frames  
**What it does**: Extracts and analyzes individual frames from candidate scenes
- Selects representative frames from each candidate scene
- Applies image processing to enhance text visibility
- Prepares frames for OCR processing

**When to use**: After Step 1, and after reviewing/selecting which scenes to process

**⚠️ Review Candidate Scenes**: Before running Step 2, you can review the detected scenes and deselect any that clearly don't contain credits to save processing time.

### Step 3: Azure VLM OCR
**What it does**: Extracts text from credit frames using advanced OCR
- Uses Azure's Vision Language Model for high-accuracy text extraction
- Identifies person names, role groups (Actor, Director, Producer, etc.)
- Associates extracted text with specific video frames

**When to use**: After Step 2 completes successfully

### Step 4: IMDB Validation  
**What it does**: Cross-references extracted names with IMDB database
- Validates that extracted names are real people/companies
- Marks entries as "Person" or "Company"
- Flags potential issues for manual review

**When to use**: After Step 3 to validate and clean up results

### Run All Steps
Executes all four steps sequentially for hands-off processing.

---

## ✏️ Review & Edit Credits

### Accessing the Review Interface
1. Click the **✏️ Review & Edit Credits** tab
2. Select an episode from the dropdown
3. Choose filtering options:
   - **Show all episodes**: Displays all processed episodes with issue counts
   - **Show only episodes that need reviewing**: Filters to episodes with problematic credits

### Understanding Credit Issues
Credits may be flagged for review due to:
- **IMDB Not Found**: Name not found in IMDB database
- **Duplicate Entries**: Same person appears multiple times with different details
- **Ambiguous Role**: Unclear or inconsistent role information
- **Low Confidence**: OCR confidence below threshold

### Review Interface

#### Progress Tracking
- **Current**: Shows which credit you're reviewing (e.g., "1/15")
- **✅ Resolved**: Number of credits you've processed  
- **🗑️ Deleted**: Number of credits marked for deletion
- **⏳ Remaining**: Credits still needing review

#### Credit Information Display
For each problematic credit, you'll see:
- **Name**: The extracted person/company name
- **Issues**: Specific problems detected
- **Role Group**: Category (Actor, Director, Producer, etc.)
- **Role Detail**: Specific role or character name
- **Source Frames**: Images where the name was found

### Editing Actions

#### For Single Credits
- **Edit Name**: Correct spelling or formatting
- **Change Role Group**: Select from dropdown of available roles
- **Edit Role Detail**: Add or modify specific role information
- **Change Type**: Switch between "Person" and "Company"
- **Delete**: Remove the credit entirely

#### For Duplicate Credits
When a person appears multiple times:
- **Compare Variants**: View all instances side-by-side
- **Edit Each Variant**: Modify details for each occurrence
- **Delete Variants**: Remove duplicate or incorrect entries
- **Add New Entries**: Create additional credits for the same person

#### Adding New Credits
1. Click **➕ Add New Entry**
2. Fill in name, role group, and role detail
3. Select whether it's a Person or Company
4. Choose source frame if available
5. Click **💾 Save All Changes**

### Navigation
- **⏮️ Previous**: Go to previous problematic credit
- **⏭️ Next**: Go to next problematic credit  
- **Skip**: Skip current credit without changes
- **Auto-navigate**: Automatically moves to next credit after saving changes

### Batch Operations
- **💾 Save All Changes**: Applies all edits for current credit
- **🔄 Refresh Episodes**: Updates the episode list and counts

---

## 🎯 Best Practices

### Before Processing
1. **Organize Videos**: Place all videos in `data/raw/` folder
2. **Configure Language**: Set OCR language to match video content
3. **Add Stopwords**: Include channel logos and common watermarks
4. **Choose Detection Method**: Use scene count for most content

### During Processing  
1. **Start with Step 1**: Always begin with scene detection
2. **Review Candidate Scenes**: Deselect obvious non-credit scenes in Step 2 review
3. **Monitor Progress**: Check logs for any errors or warnings
4. **Be Patient**: OCR processing can take time depending on video length

### During Review
1. **Focus on Issues**: Use "Show only episodes that need reviewing" filter
2. **Verify Names**: Check that extracted names make sense in context
3. **Consistent Roles**: Ensure role groups are standardized
4. **Delete Duplicates**: Remove redundant or incorrect entries
5. **Save Regularly**: Use "Save All Changes" frequently

### Quality Control
- **Cross-reference**: Verify names against cast lists when possible
- **Role Accuracy**: Ensure role groups match actual functions
- **Completeness**: Check that major roles are captured
- **Consistency**: Use standardized naming and role conventions

---

## 🔧 Troubleshooting

### Common Issues

#### "No videos found"
- Verify videos are in `data/raw/` folder
- Check file formats (must be `.mp4`, `.mkv`, `.avi`, or `.mov`)
- Ensure filenames don't contain special characters

#### "OCR errors" or poor text extraction
- Try clicking **🔄 Refresh OCR Reader** 
- Verify correct language is selected
- Check that video quality is sufficient for text recognition

#### "Scene detection finds no credit scenes"
- Try different detection method (time-based vs scene count)
- Increase scene count or time duration margins
- Consider using "Whole Episode" for unusual content

#### "Too many false positives in scene detection"
- Reduce scene count or time duration
- Add more stopwords for channel logos/watermarks
- Review and deselect obvious non-credit scenes before Step 2

#### "IMDB validation marks everything as not found"
- Names may be correctly extracted but not in IMDB database
- Check for spelling variations in extracted names
- Some legitimate credits may not be in IMDB (crew, local talent)

### Performance Tips
- **GPU Acceleration**: If available, GPU processing is significantly faster
- **Scene Selection**: Deselecting obvious non-credit scenes speeds up processing
- **Batch Processing**: Process multiple episodes in sequence
- **Review Efficiently**: Use filters to focus only on episodes needing attention

---

## 📁 Output and Data

### Database Storage
Extracted credits are stored in `db/tvcredits_v3.db` (SQLite database)

### File Organization
```
FilmoCredit/
├── data/
│   ├── raw/                    # Your input videos
│   └── episodes/              # Processing data per episode
│       └── [Episode Name]/
│           ├── analysis/      # Scene detection results
│           └── ocr/          # OCR extraction results
├── db/                       # Database files
└── logs/                     # Processing logs
```

### Accessing Results
- Use the **Review & Edit Credits** tab to view and export results
- Database can be accessed with SQLite tools for advanced analysis
- Processing logs available in **📊 Logs** tab

---

## 🎉 Success Tips

1. **Start Small**: Test with one short video first
2. **Iterate**: Adjust settings based on initial results
3. **Review Thoroughly**: Manual review significantly improves accuracy
4. **Standardize**: Use consistent naming and role conventions
5. **Document**: Keep notes on optimal settings for different content types

FilmoCredit combines powerful AI automation with human oversight to deliver accurate, comprehensive credit extraction. The key to success is finding the right balance between automated processing and manual review for your specific content.

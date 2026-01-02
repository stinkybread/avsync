# AVSync Desktop - UI Redesign Implementation Plan

## Completed:
✅ Theme colors updated (pink/magenta accent)
✅ Minimum window size set to 1800x900
✅ JobQueue component created
✅ SyncPointPills component created (horizontal pill design)

## Remaining Changes:

### 1. Fix TypeScript Error in JobQueue.tsx
Remove unused import:
```typescript
// Line 1: Remove 'useState' from import
import './JobQueue.css'
```

### 2. Update App.tsx - Complete Redesign

**New State:**
```typescript
- Add job queue state: `const [jobs, setJobs] = useState<Job[]>([])`
- Add current job ID: `const [currentJobId, setCurrentJobId] = useState<string | null>(null)`
- Keep existing state for creating new jobs
```

**New Layout (3-column grid):**
```css
.app-content {
  display: grid;
  grid-template-columns: 380px 1fr 420px;
  gap: 1px;
  background-color: var(--border-color);
}
```

**Column Structure:**
- **Left (380px)**: FileSelector + ParametersPanel
- **Middle (flex)**: VideoPreview (top 60%) + LogViewer (bottom 40%)
- **Right (420px)**: JobQueue (top 50%) + SyncPointPills (bottom 50%)

### 3. Update Header.tsx

Change button from "Run AVSync" to "Add to Queue":
```typescript
<button className="btn btn-primary" onClick={onAddToQueue}>
  Add to Queue
</button>
```

### 4. Update VideoPreview Component

**Remove:**
- Bottom sync point action (move to separate component)
- SyncPointEditor integration

**Keep:**
- Side-by-side video frames
- Navigation controls
- Timeline sliders
- "Add Sync Point" button (now calls parent callback)

**New CSS:**
```css
.video-preview {
  height: 100%;
  display: flex;
  flex-direction: column;
}
```

### 5. Update LogViewer Component

**Changes:**
- Show logs for current job only
- Add job selector if multiple jobs
- Auto-switch to processing job

### 6. Create New App.css

```css
.app {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-content {
  flex: 1;
  display: grid;
  grid-template-columns: 380px 1fr 420px;
  gap: 1px;
  background-color: var(--border-color);
  overflow: hidden;
}

.left-column {
  display: flex;
  flex-direction: column;
  background-color: var(--bg-primary);
  overflow-y: auto;
}

.middle-column {
  display: flex;
  flex-direction: column;
  background-color: var(--bg-primary);
  overflow: hidden;
}

.middle-top {
  flex: 0 0 60%;
  overflow-y: auto;
}

.middle-bottom {
  flex: 1;
  border-top: 1px solid var(--border-color);
}

.right-column {
  display: flex;
  flex-direction: column;
  background-color: var(--bg-primary);
  gap: 1px;
  overflow: hidden;
}

.right-top {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.right-bottom {
  flex: 0 0 auto;
  max-height: 50%;
}
```

### 7. Job Queue Integration

**Add to Queue Logic:**
```typescript
const handleAddToQueue = () => {
  if (!refVideo || !foreignVideo || !outputPath) {
    alert('Please select videos and output path');
    return;
  }

  const newJob: Job = {
    id: `job-${Date.now()}`,
    refVideo: refVideo.path,
    foreignVideo: foreignVideo.path,
    outputVideo: outputPath,
    status: 'pending',
    logs: [],
    createdAt: new Date(),
    params: {
      ...parameters,
      forceSyncPoints: syncPoints.length > 0
        ? syncPoints.map(sp => `${sp.refTime}:${sp.foreignTime}`).join(',')
        : undefined
    }
  };

  setJobs([...jobs, newJob]);

  // Start processing if no job is running
  if (!currentJobId) {
    processNextJob([...jobs, newJob]);
  }
};
```

**Process Jobs:**
```typescript
const processNextJob = (jobList: Job[]) => {
  const nextJob = jobList.find(j => j.status === 'pending');
  if (!nextJob) return;

  setCurrentJobId(nextJob.id);
  updateJobStatus(nextJob.id, 'processing');

  // Call existing runAVSync logic
  window.electronAPI.runAVSync({
    refVideo: nextJob.refVideo,
    foreignVideo: nextJob.foreignVideo,
    outputVideo: nextJob.outputVideo,
    params: nextJob.params
  });
};
```

### 8. Component Imports in App.tsx

```typescript
import Header from './components/Header'
import FileSelector from './components/FileSelector'
import VideoPreview from './components/VideoPreview'
import ParametersPanel from './components/ParametersPanel'
import LogViewer from './components/LogViewer'
import JobQueue, { Job, JobStatus } from './components/JobQueue'
import SyncPointPills from './components/SyncPointPills'
```

### 9. Update Package Colors

Update all components to use new accent colors:
- Primary actions → `var(--accent-primary)` (pink)
- Success → `var(--accent-green)`
- Error → `var(--accent-red)`
- Info/Blue → `var(--accent-blue)`
- Warning → `var(--accent-yellow)`

### 10. Font Size Adjustments

Already set in index.css:
```css
:root {
  font-size: 14px; /* Base for 1800x900 */
}
```

All components use `rem` or relative units, so they'll scale appropriately.

## Implementation Order:

1. Fix JobQueue.tsx TypeScript error
2. Update App.tsx with new layout and job queue logic
3. Update VideoPreview.tsx to remove sync point editor
4. Update LogViewer.tsx to show current job logs
5. Test job queue functionality
6. Adjust spacing/sizes as needed

## Testing Checklist:

- [ ] Window opens at 1800x900
- [ ] 3-column layout displays correctly
- [ ] File selection works
- [ ] Parameters can be adjusted
- [ ] Video preview loads frames
- [ ] Sync points can be added/edited/removed (pills)
- [ ] Jobs can be added to queue
- [ ] Jobs process in order
- [ ] Logs show for current job
- [ ] Jobs can be retried/removed
- [ ] Theme colors look correct (pink accent)

## Quick Start:

The files are created and ready. To complete:
1. Fix the import in JobQueue.tsx
2. Replace App.tsx content with new 3-column layout
3. Run `npm run build && npm run dev`

Would you like me to provide the complete App.tsx file with all the changes integrated?

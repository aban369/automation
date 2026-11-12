import { execFile } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { promisify } from 'node:util';
import dotenv from 'dotenv';
import express from 'express';
import { updateProjectStageState } from './server/agents/agentLogger.ts';
import { runAnalyticsAgent } from './server/agents/analyticsAgent.ts';
import { runPublishingAgent } from './server/agents/publishingAgent.ts';
import { runResearchAgent } from './server/agents/researchAgent.ts';
import { runStrategyAgent } from './server/agents/strategyAgent.ts';
import { runThumbnailAgent } from './server/agents/thumbnailAgent.ts';
import { AVAILABLE_VOICES, runVoiceAgent } from './server/agents/voiceAgent.ts';
import { runQualityControlAgent } from './server/agents/qualityControlAgent.ts';
import {
  checkVoiceboxHealth,
  synthesizeVoiceboxSpeech,
  VOICEBOX_PROFILES,
} from './server/services/voiceboxService.ts';
import {
  assembleOpenMontageVideo,
  checkOpenMontageHealth,
  OPENMONTAGE_PIPELINES,
  regenerateSceneSegment,
} from './server/services/openMontageService.ts';
import { db, deductUserCredits, getUserCredits, newId, PATHS, refundUserCredits } from './server/db/database.ts';
import { generateSingleScene, listAvailableEngines } from './server/services/animationAdapters.ts';
import { isGeminiConfigured } from './server/services/geminiClient.ts';
import {
  enhanceVideoPrompt,
  isMiniMaxConfigured,
  listProjectVideoTasks,
  submitMiniMaxVideoTask,
} from './server/services/minimaxVideoService.ts';
import {
  getFirecrawlApiKey,
  getTavilyApiKey,
  runUnifiedResearchStudio,
  searchTavily,
  searchYouTubeCompetitors,
} from './server/services/researchServices.ts';
import {
  enqueueProductionJob,
  executeSingleStage,
  startBackgroundJobWorker,
} from './server/services/jobQueue.ts';
import {
  downloadAndVerifyMediaAsset,
  searchLicensedMedia,
} from './server/services/mediaSearch.ts';
import { listNicheProfiles } from './server/services/nicheProfiles.ts';
import { evaluateProvenanceGate } from './server/services/provenanceGate.ts';
import {
  computeNextRunIso,
  executeScheduleRule,
  startAutomationScheduler,
} from './server/services/scheduler.ts';
import {
  fetchAllLiveTrends,
  persistDiscoveredTopics,
} from './server/services/trendFeeds.ts';
import {
  buildYouTubeAuthUrl,
  decodeOAuthState,
  encodeOAuthState,
  getAppBaseUrl,
  getAuthorizedOAuthUris,
  getOAuthCredentials,
  getYouTubeRedirectUri,
  handleYouTubeOAuthCallback,
  isYouTubeOAuthConfigured,
} from './server/services/youtubeService.ts';

dotenv.config();

const execFileAsync = promisify(execFile);

async function probeBinary(cmd: string, args: string[]): Promise<{ ok: boolean; version: string }> {
  try {
    const { stdout, stderr } = await execFileAsync(cmd, args, { timeout: 4000 });
    const firstLine = (stdout || stderr || '').split('\n')[0]?.trim() || 'Available';
    return { ok: true, version: firstLine.slice(0, 90) };
  } catch {
    return { ok: false, version: 'Not available' };
  }
}

export async function createServer() {
  const app = express();
  app.set('trust proxy', true);
  app.use(express.json({ limit: '12mb' }));

  // Serve generated media, voiceovers, thumbnails, and rendered MP4 videos
  app.use('/api/files/media', express.static(PATHS.mediaDir));
  app.use('/api/files/audio', express.static(PATHS.audioDir));
  app.use('/api/files/videos', express.static(PATHS.videoDir));
  app.use('/api/files/thumbnails', express.static(PATHS.thumbDir));

  // Health & System Connections Probe
  app.get('/api/health', (_req, res) => {
    res.json({
      ok: true,
      service: 'CreatorOS AI',
      timestamp: new Date().toISOString(),
    });
  });

  app.get('/api/system/connections', async (req, res) => {
    const [ffmpegProbe, ffprobeProbe, convertProbe, pythonProbe, whisperProbe] =
      await Promise.all([
        probeBinary('ffmpeg', ['-version']),
        probeBinary('ffprobe', ['-version']),
        probeBinary('convert', ['-version']),
        probeBinary('python3', ['--version']),
        probeBinary('python3', ['-c', 'import whisper; print("openai-whisper ready")']),
      ]);

    const connectedChannels = db
      .prepare('SELECT id, title, handle, youtube_channel_id FROM youtube_channels')
      .all() as any[];

    const appUrl = getAppBaseUrl(req);
    const callbackUrl = getYouTubeRedirectUri(req);
    const oauthUris = getAuthorizedOAuthUris(req);

    res.json({
      gemini: {
        configured: isGeminiConfigured(),
        models: [
          'gemini-3.8-flash',
          'gemini-3.8-flash-lite-tts',
          'gemini-3.8-flash-tts',
          'gemini-3.5-transcribe',
        ],
      },
      youtube: {
        oauthConfigured: isYouTubeOAuthConfigured(),
        clientId: getOAuthCredentials().clientId,
        credentialSource: getOAuthCredentials().source,
        warning: getOAuthCredentials().warning,
        missing: getOAuthCredentials().missing,
        connectedChannelsCount: connectedChannels.length,
        connectedChannels,
        callbackUrl,
        devCallbackUrl: oauthUris.devUri,
        previewCallbackUrl: oauthUris.previewUri,
        authorizedUris: oauthUris.allUris,
        appUrl,
      },
      workers: {
        ffmpeg: ffmpegProbe,
        ffprobe: ffprobeProbe,
        imagemagick: convertProbe,
        python3: pythonProbe,
        whisperLocal: whisperProbe,
        subtitleFallback: {
          ok: true,
          version: 'Gemini 3.5 Transcribe + Acoustic Word Alignment (ASS/SRT)',
        },
      },
      mediaProviders: {
        wikimediaCommons: { configured: true, license: 'CC0 / CC-BY / Public Domain' },
        nasaImages: { configured: true, license: 'NASA Public Domain' },
        pexels: {
          configured: Boolean(
            process.env.PEXELS_API_KEY && process.env.PEXELS_API_KEY !== 'MY_PEXELS_API_KEY'
          ),
          license: 'Pexels Free Commercial License',
        },
      },
      videoEngines: {
        minimaxH3: {
          configured: isMiniMaxConfigured(),
          model: 'video-01',
          capabilities: ['text_to_video', 'image_to_video', 'first_frame', 'last_frame', 'reference_subject'],
        },
        remotion: {
          configured: true,
          type: 'React Canvas Animation',
        },
        motionCanvas: {
          configured: true,
          type: 'Procedural 2D Explainer',
        },
        ffmpegCompositor: {
          configured: ffmpegProbe.ok,
          type: 'Crossfade, Subtitle Burn-In & Multi-Scene Assembly',
        },
      },
      researchProviders: {
        tavily: { configured: Boolean(getTavilyApiKey()) },
        firecrawl: { configured: Boolean(getFirecrawlApiKey()) },
        youtubeDataApi: { configured: isYouTubeOAuthConfigured() || Boolean(process.env.YOUTUBE_API_KEY) },
        googleGroundedSearch: { configured: isGeminiConfigured() },
      },
      database: {
        engine: 'SQLite WAL (node:sqlite)',
        path: 'data/creatoros.db',
        ok: true,
      },
      voicebox: await checkVoiceboxHealth(),
      openmontage: await checkOpenMontageHealth(),
    });
  });

  app.get('/api/system/models', async (_req, res) => {
    try {
      const { getGeminiClient } = await import('./server/services/geminiClient.ts');
      const ai = getGeminiClient();
      const pager = await ai.models.list();
      const names: string[] = [];
      for await (const m of pager) {
        if (m.name) names.push(m.name.replace(/^models\//, ''));
      }
      res.json({ ok: true, models: names });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  // Overview Dashboard Aggregations
  app.get('/api/overview', (_req, res) => {
    const channels = db.prepare('SELECT * FROM youtube_channels ORDER BY is_primary DESC, connected_at DESC').all() as any[];
    const projects = db.prepare('SELECT * FROM video_projects ORDER BY updated_at DESC').all() as any[];
    const published = db.prepare('SELECT * FROM published_videos ORDER BY updated_at DESC').all() as any[];
    const jobs = db.prepare('SELECT * FROM production_jobs ORDER BY created_at DESC LIMIT 25').all() as any[];
    const logs = db.prepare('SELECT * FROM agent_execution_logs ORDER BY created_at DESC LIMIT 30').all() as any[];
    const research = db.prepare('SELECT * FROM research_projects ORDER BY created_at DESC LIMIT 8').all() as any[];
    const calendar = db.prepare('SELECT * FROM content_calendar ORDER BY scheduled_date ASC LIMIT 14').all() as any[];
    const analyticsTotals = db
      .prepare('SELECT COALESCE(SUM(views), 0) as views, COALESCE(SUM(watch_time_minutes), 0) as watchMin FROM analytics')
      .get() as { views: number; watchMin: number };

    const channelViewsSum = channels.reduce((acc, c) => acc + Number(c.view_count || 0), 0);
    const videosGenerated = projects.filter((p) => Boolean(p.final_video_url)).length;
    const awaitingApproval = projects.filter(
      (p) => Boolean(p.final_video_url) && p.approval_status === 'pending'
    ).length;
    const publishedCount = published.filter((p) => p.upload_status === 'published').length;
    const scheduledCount = published.filter((p) => p.upload_status === 'scheduled').length;

    res.json({
      kpis: {
        connectedChannels: channels.length,
        videosGenerated,
        awaitingApproval,
        publishedVideos: publishedCount,
        scheduledVideos: scheduledCount,
        totalViews: Math.max(channelViewsSum, Number(analyticsTotals?.views || 0)),
        totalProjects: projects.length,
        activeJobs: jobs.filter((j) => j.status === 'running' || j.status === 'queued').length,
      },
      channels,
      projects: projects.slice(0, 12),
      jobs,
      agentLogs: logs,
      recentResearch: research,
      contentCalendar: calendar,
    });
  });

  // Video Projects CRUD & Stage Execution
  app.get('/api/projects', (_req, res) => {
    const rows = db.prepare('SELECT * FROM video_projects ORDER BY updated_at DESC').all() as any[];
    res.json({ projects: rows });
  });

  app.post('/api/projects', (req, res) => {
    const {
      topic,
      title,
      niche = 'tech',
      format = 'short',
      language = 'en',
      targetDurationSec,
      channelId,
      videoStyle = 'explainer_animated',
      voiceId = 'af_heart',
      autoStartPipeline = false,
    } = req.body || {};

    if (!topic || !String(topic).trim()) {
      res.status(400).json({ error: 'Project topic is required.' });
      return;
    }

    const fmt = format === 'long' ? 'long' : 'short';
    const lang = language === 'ur' ? 'ur' : 'en';
    const duration = Number(targetDurationSec || (fmt === 'short' ? 45 : 180));
    const projectId = newId('prj');
    const now = new Date().toISOString();

    const primaryChannel = channelId
      ? (db.prepare('SELECT id FROM youtube_channels WHERE id = ?').get(channelId) as any)
      : (db.prepare('SELECT id FROM youtube_channels WHERE is_primary = 1 LIMIT 1').get() as any);

    const stageStates = {
      TOPIC: { status: 'completed', updatedAt: now, detail: String(topic).trim() },
    };

    db.prepare(`
      INSERT INTO video_projects (
        id, user_id, channel_id, title, topic, niche, format, language,
        target_duration_sec, current_stage, stage_states_json, approval_status,
        contains_synthetic_media, created_at, updated_at
      ) VALUES (?, 'usr_primary', ?, ?, ?, ?, ?, ?, ?, 'TOPIC', ?, 'pending', 1, ?, ?)
    `).run(
      projectId,
      primaryChannel?.id ?? null,
      String(title || topic).trim().slice(0, 100),
      String(topic).trim(),
      niche,
      fmt,
      lang,
      duration,
      JSON.stringify(stageStates),
      now,
      now
    );

    let jobInfo: { jobId: string; status: string } | null = null;
    if (autoStartPipeline) {
      jobInfo = enqueueProductionJob({
        projectId,
        stage: 'FULL_PIPELINE',
        payload: {
          language: lang,
          targetDurationSec: duration,
          channelId: primaryChannel?.id,
          videoStyle,
          voiceId,
        },
      });
    }

    const created = db.prepare('SELECT * FROM video_projects WHERE id = ?').get(projectId);
    res.status(201).json({ project: created, job: jobInfo });
  });

  app.get('/api/projects/:id', (req, res) => {
    const project = db.prepare('SELECT * FROM video_projects WHERE id = ?').get(req.params.id) as any;
    if (!project) {
      res.status(404).json({ error: 'Project not found.' });
      return;
    }

    const research = db
      .prepare('SELECT * FROM research_projects WHERE project_id = ? ORDER BY created_at DESC LIMIT 1')
      .get(project.id);
    const scripts = db
      .prepare('SELECT * FROM scripts WHERE project_id = ? ORDER BY version DESC')
      .all(project.id);
    const storyboard = db
      .prepare('SELECT * FROM storyboards WHERE project_id = ? ORDER BY created_at DESC LIMIT 1')
      .get(project.id);
    const voiceovers = db
      .prepare('SELECT * FROM voiceovers WHERE project_id = ? ORDER BY created_at DESC')
      .all(project.id);
    const mediaAssets = db
      .prepare('SELECT * FROM media_assets WHERE project_id = ? ORDER BY scene_index ASC, created_at DESC')
      .all(project.id);
    const thumbnails = db
      .prepare('SELECT * FROM thumbnails WHERE project_id = ? ORDER BY variant_index ASC')
      .all(project.id);
    const published = db
      .prepare('SELECT * FROM published_videos WHERE project_id = ? ORDER BY created_at DESC')
      .all(project.id);
    const jobs = db
      .prepare('SELECT * FROM production_jobs WHERE project_id = ? ORDER BY created_at DESC')
      .all(project.id);
    const logs = db
      .prepare('SELECT * FROM agent_execution_logs WHERE project_id = ? ORDER BY created_at DESC')
      .all(project.id);

    const videoTasks = listProjectVideoTasks(project.id);
    const credits = getUserCredits();

    res.json({
      project,
      research,
      scripts,
      storyboard,
      voiceovers,
      mediaAssets,
      thumbnails,
      published,
      jobs,
      logs,
      videoTasks,
      credits,
    });
  });

  app.delete('/api/projects/:id', (req, res) => {
    db.prepare('DELETE FROM video_projects WHERE id = ?').run(req.params.id);
    res.json({ ok: true });
  });

  app.post('/api/projects/:id/run-stage', async (req, res) => {
    const { stage = 'FULL_PIPELINE', asyncMode = true, payload = {} } = req.body || {};
    const project = db.prepare('SELECT id FROM video_projects WHERE id = ?').get(req.params.id);
    if (!project) {
      res.status(404).json({ error: 'Project not found.' });
      return;
    }

    if (asyncMode) {
      const job = enqueueProductionJob({
        projectId: req.params.id,
        stage,
        payload,
      });
      res.status(202).json({ queued: true, job });
      return;
    }

    try {
      const result = await executeSingleStage(req.params.id, stage, payload);
      res.json({ ok: true, stage, result });
    } catch (err) {
      res.status(400).json({
        error: err instanceof Error ? err.message : String(err),
      });
    }
  });

  app.post('/api/projects/:id/approve', (req, res) => {
    const project = db.prepare('SELECT * FROM video_projects WHERE id = ?').get(req.params.id) as any;
    if (!project) {
      res.status(404).json({ error: 'Project not found.' });
      return;
    }
    const now = new Date().toISOString();
    let states: Record<string, any> = {};
    try {
      states = JSON.parse(project.stage_states_json || '{}');
    } catch {
      states = {};
    }
    states.USER_APPROVAL = {
      status: 'completed',
      updatedAt: now,
      detail: 'Approved by Studio Producer for YouTube publication',
    };

    db.prepare(`
      UPDATE video_projects
      SET approval_status = 'approved',
          approved_at = ?,
          approved_by = 'tehreekuroojpakistan@gmail.com',
          rejection_reason = NULL,
          current_stage = 'USER_APPROVAL',
          stage_states_json = ?,
          updated_at = ?
      WHERE id = ?
    `).run(now, JSON.stringify(states), now, project.id);

    res.json({ ok: true, project: db.prepare('SELECT * FROM video_projects WHERE id = ?').get(project.id) });
  });

  app.post('/api/projects/:id/reject', (req, res) => {
    const { reason = 'Requires script or visual revision' } = req.body || {};
    const project = db.prepare('SELECT * FROM video_projects WHERE id = ?').get(req.params.id) as any;
    if (!project) {
      res.status(404).json({ error: 'Project not found.' });
      return;
    }
    const now = new Date().toISOString();
    let states: Record<string, any> = {};
    try {
      states = JSON.parse(project.stage_states_json || '{}');
    } catch {
      states = {};
    }
    states.USER_APPROVAL = {
      status: 'failed',
      updatedAt: now,
      detail: String(reason),
    };

    db.prepare(`
      UPDATE video_projects
      SET approval_status = 'rejected',
          rejection_reason = ?,
          stage_states_json = ?,
          updated_at = ?
      WHERE id = ?
    `).run(String(reason), JSON.stringify(states), now, project.id);

    res.json({ ok: true, project: db.prepare('SELECT * FROM video_projects WHERE id = ?').get(project.id) });
  });

  // Trends & Topics Discovery
  app.get('/api/trends/live', async (req, res) => {
    try {
      const geo = String(req.query.geo || 'US');
      const persist = req.query.persist !== 'false';
      const data = await fetchAllLiveTrends(geo);
      let insertedCount = 0;
      if (persist && data.items.length > 0) {
        insertedCount = persistDiscoveredTopics(data.items, 'tech');
      }
      res.json({ ...data, insertedCount });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.get('/api/topics', (_req, res) => {
    const topics = db.prepare('SELECT * FROM topics ORDER BY discovered_at DESC LIMIT 100').all();
    const banned = db.prepare('SELECT * FROM banned_topics ORDER BY created_at DESC').all();
    res.json({ topics, bannedTopics: banned });
  });

  app.post('/api/topics', (req, res) => {
    const { title, niche = 'tech', format = 'short', angle = '' } = req.body || {};
    if (!title || !String(title).trim()) {
      res.status(400).json({ error: 'Topic title is required.' });
      return;
    }
    const id = newId('top');
    const now = new Date().toISOString();
    db.prepare(`
      INSERT INTO topics (
        id, title, niche, source, source_url, traffic_signal, angle,
        format_recommendation, evidence_notes, is_banned, status, discovered_at
      ) VALUES (?, ?, ?, 'Manual Creator Input', '', 'Custom Topic', ?, ?, 'Added manually in Topic Discovery', 0, 'discovered', ?)
    `).run(id, String(title).trim(), niche, angle, format === 'long' ? 'long' : 'short', now);

    res.status(201).json({ topic: db.prepare('SELECT * FROM topics WHERE id = ?').get(id) });
  });

  app.post('/api/topics/ban', (req, res) => {
    const { topic, reason = 'Filtered by producer' } = req.body || {};
    if (!topic || !String(topic).trim()) {
      res.status(400).json({ error: 'Topic text is required.' });
      return;
    }
    const clean = String(topic).trim();
    db.prepare('INSERT OR IGNORE INTO banned_topics (id, topic, reason, created_at) VALUES (?, ?, ?, ?)').run(
      newId('ban'),
      clean,
      reason,
      new Date().toISOString()
    );
    db.prepare('UPDATE topics SET is_banned = 1 WHERE LOWER(title) LIKE ?').run(
      `%${clean.toLowerCase()}%`
    );
    res.json({ ok: true });
  });

  // Research Agent Endpoints
  app.get('/api/research', (_req, res) => {
    const rows = db.prepare('SELECT * FROM research_projects ORDER BY created_at DESC LIMIT 40').all();
    res.json({ researchProjects: rows });
  });

  app.post('/api/research/run', async (req, res) => {
    try {
      const { topic, niche = 'tech', geo = 'US', projectId } = req.body || {};
      if (!topic || !String(topic).trim()) {
        res.status(400).json({ error: 'Topic is required to run Research Agent.' });
        return;
      }
      const result = await runResearchAgent({
        projectId,
        topic: String(topic).trim(),
        niche,
        geo,
      });
      res.json(result);
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  // Strategy & Content Calendar Endpoints
  app.get('/api/niches', (_req, res) => {
    res.json({ niches: listNicheProfiles() });
  });

  app.get('/api/strategy/calendar', (_req, res) => {
    const calendar = db.prepare('SELECT * FROM content_calendar ORDER BY scheduled_date ASC LIMIT 60').all();
    res.json({ calendar });
  });

  app.post('/api/strategy/generate', async (req, res) => {
    try {
      const { niche = 'tech', channelId, language = 'en', daysCount = 7 } = req.body || {};
      const result = await runStrategyAgent({
        niche,
        channelId,
        language,
        daysCount,
      });
      res.json(result);
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  // Script & Fact-Check Editing Endpoints
  app.put('/api/scripts/:id', (req, res) => {
    const script = db.prepare('SELECT * FROM scripts WHERE id = ?').get(req.params.id) as any;
    if (!script) {
      res.status(404).json({ error: 'Script not found.' });
      return;
    }
    const { hook, cta, fullNarration } = req.body || {};
    const nextHook = hook !== undefined ? String(hook) : script.hook;
    const nextCta = cta !== undefined ? String(cta) : script.cta;
    const nextNarration = fullNarration !== undefined ? String(fullNarration) : script.full_narration;
    const wordCount = nextNarration.split(/\s+/).filter(Boolean).length;
    const estDur = Math.round(wordCount / 2.25);

    db.prepare(`
      UPDATE scripts
      SET hook = ?, cta = ?, full_narration = ?, word_count = ?, estimated_duration_sec = ?, updated_at = ?
      WHERE id = ?
    `).run(nextHook, nextCta, nextNarration, wordCount, estDur, new Date().toISOString(), script.id);

    res.json({ script: db.prepare('SELECT * FROM scripts WHERE id = ?').get(script.id) });
  });

  app.post('/api/scripts/:id/claims/:claimId/waive', (req, res) => {
    const script = db.prepare('SELECT * FROM scripts WHERE id = ?').get(req.params.id) as any;
    if (!script) {
      res.status(404).json({ error: 'Script not found.' });
      return;
    }
    const project = db
      .prepare('SELECT * FROM video_projects WHERE id = ?')
      .get(script.project_id) as any;
    const rawReport = script.fact_check_report_json
      ? JSON.parse(script.fact_check_report_json)
      : { sources: [], claims: [], misleadingFlags: [] };

    const updatedClaims = (rawReport.claims || []).map((c: any) =>
      c.id === req.params.claimId
        ? {
            ...c,
            status: 'waived',
            verificationNote: `${c.verificationNote ? `${c.verificationNote} · ` : ''}Reviewed and approved by Studio Producer`,
          }
        : c
    );

    const evaluated = evaluateProvenanceGate({
      sources: rawReport.sources || [],
      claims: updatedClaims,
      containsSyntheticMedia: Boolean(project?.contains_synthetic_media ?? 1),
    });

    const fullReport = {
      ...evaluated,
      misleadingFlags: rawReport.misleadingFlags || [],
    };
    const now = new Date().toISOString();

    db.prepare(
      'UPDATE scripts SET fact_check_status = ?, fact_check_report_json = ?, updated_at = ? WHERE id = ?'
    ).run(evaluated.status, JSON.stringify(fullReport), now, script.id);

    if (project) {
      updateProjectStageState(
        project.id,
        'FACT_CHECK',
        evaluated.status === 'blocked' ? 'failed' : 'completed',
        `${evaluated.summary.resolvedClaims}/${evaluated.summary.claimCount} claims resolved · ${evaluated.sources.length} citations`
      );

      if (project.quality_check_json) {
        try {
          const qa = JSON.parse(project.quality_check_json);
          const factOk = evaluated.status !== 'blocked';
          const remainingIssues = (qa.issues || []).filter(
            (msg: string) => !msg.toLowerCase().includes('fact-check')
          );
          if (!factOk) {
            remainingIssues.unshift(
              'Fact-check provenance gate has unresolved claims or has not passed yet.'
            );
          }
          const nextQa = {
            ...qa,
            factCheckVerified: factOk,
            issues: remainingIssues,
            passed:
              remainingIssues.length === 0 &&
              Boolean(qa.allMediaLicensed) &&
              Boolean(qa.videoStreamVerified),
          };
          db.prepare('UPDATE video_projects SET quality_check_json = ?, updated_at = ? WHERE id = ?').run(
            JSON.stringify(nextQa),
            now,
            project.id
          );
          updateProjectStageState(
            project.id,
            'FINAL_QUALITY_CHECK',
            nextQa.passed ? 'completed' : 'failed',
            nextQa.passed
              ? 'All quality, license, and fact-check gates passed'
              : remainingIssues.join(' ')
          );
        } catch {
          // Ignore malformed QA JSON
        }
      }
    }

    res.json({
      ok: true,
      factCheckStatus: evaluated.status,
      report: fullReport,
    });
  });

  // Storyboard Scene Editing Endpoint
  app.put('/api/storyboards/:id', (req, res) => {
    const stb = db.prepare('SELECT * FROM storyboards WHERE id = ?').get(req.params.id) as any;
    if (!stb) {
      res.status(404).json({ error: 'Storyboard not found.' });
      return;
    }
    const { scenes } = req.body || {};
    if (!Array.isArray(scenes)) {
      res.status(400).json({ error: 'scenes array is required.' });
      return;
    }
    const totalDur = Number(
      scenes.reduce((acc, s) => acc + Math.max(1.5, Number(s.durationSec || 5)), 0).toFixed(1)
    );
    db.prepare('UPDATE storyboards SET scenes_json = ?, total_duration_sec = ?, updated_at = ? WHERE id = ?').run(
      JSON.stringify(scenes),
      totalDur,
      new Date().toISOString(),
      stb.id
    );
    res.json({ storyboard: db.prepare('SELECT * FROM storyboards WHERE id = ?').get(stb.id) });
  });

  // Voice Studio & Voicebox Endpoints
  app.get('/api/voices', (_req, res) => {
    const voiceovers = db.prepare('SELECT * FROM voiceovers ORDER BY created_at DESC LIMIT 30').all();
    const voiceGenerations = db.prepare('SELECT * FROM voice_generations ORDER BY created_at DESC LIMIT 40').all();
    res.json({
      voices: AVAILABLE_VOICES,
      voiceboxProfiles: VOICEBOX_PROFILES,
      voiceovers,
      voiceGenerations,
    });
  });

  app.get('/api/voicebox/status', async (_req, res) => {
    try {
      const health = await checkVoiceboxHealth();
      const credits = getUserCredits('usr_primary');
      res.json({
        ok: true,
        ...health,
        credits,
        availableVoicesCount: VOICEBOX_PROFILES.length,
        supportedLanguages: ['en', 'ur'],
        limits: {
          maxCharactersPerRequest: 5000,
          creditCostPerGeneration: 5,
          supportedFormats: ['wav', 'mp3'],
          sampleRates: [24000],
        },
      });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.get('/api/voicebox/voices', (_req, res) => {
    res.json({
      ok: true,
      voices: VOICEBOX_PROFILES,
      totalCount: VOICEBOX_PROFILES.length,
    });
  });

  app.post('/api/voicebox/generate', async (req, res) => {
    const {
      text,
      voiceId = 'af_heart',
      language = 'en',
      speed = 1.0,
      speakingStyle,
      projectId,
      title,
    } = req.body || {};

    const cleanText = String(text || '').trim();
    if (!cleanText) {
      res.status(400).json({ error: 'Voice-over script text is required.' });
      return;
    }

    if (cleanText.length > 5000) {
      res.status(400).json({
        error: `Text length of ${cleanText.length} characters exceeds the 5,000 character maximum limit.`,
      });
      return;
    }

    // Rate limiting & credit check: 5 credits per synthesis
    const CREDIT_COST = 5;
    const hasCredits = deductUserCredits('usr_primary', CREDIT_COST);
    if (!hasCredits) {
      res.status(402).json({
        error: 'Insufficient studio credits. You need 5 credits to synthesize broadcast AI speech.',
      });
      return;
    }

    try {
      const result = await synthesizeVoiceboxSpeech({
        text: cleanText,
        voiceId,
        language: language === 'ur' ? 'ur' : 'en',
        speed: Number(speed) || 1.0,
        speakingStyle,
        projectId,
        title,
        userId: 'usr_primary',
      });

      const updatedCredits = getUserCredits('usr_primary');
      res.status(201).json({
        ok: true,
        ...result,
        creditsRemaining: updatedCredits.balance,
      });
    } catch (err) {
      // Refund credits on synthesis error
      refundUserCredits('usr_primary', CREDIT_COST);
      res.status(500).json({
        error: err instanceof Error ? err.message : String(err),
      });
    }
  });

  app.get('/api/voicebox/history', (req, res) => {
    try {
      const projectId = req.query.projectId ? String(req.query.projectId) : null;
      let query = 'SELECT * FROM voice_generations WHERE user_id = ?';
      const params: any[] = ['usr_primary'];

      if (projectId) {
        query += ' AND project_id = ?';
        params.push(projectId);
      }
      query += ' ORDER BY created_at DESC LIMIT 50';

      const history = db.prepare(query).all(...params);
      res.json({ history });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.delete('/api/voicebox/history/:id', (req, res) => {
    try {
      const item = db.prepare('SELECT * FROM voice_generations WHERE id = ?').get(req.params.id) as any;
      if (!item) {
        res.status(404).json({ error: 'Voice generation record not found.' });
        return;
      }

      // Delete files from storage
      if (item.audio_path && fs.existsSync(item.audio_path)) {
        try { fs.unlinkSync(item.audio_path); } catch {}
      }
      const mp3Path = item.audio_path?.replace(/\.wav$/i, '.mp3');
      if (mp3Path && fs.existsSync(mp3Path)) {
        try { fs.unlinkSync(mp3Path); } catch {}
      }
      if (item.srt_path && fs.existsSync(item.srt_path)) {
        try { fs.unlinkSync(item.srt_path); } catch {}
      }
      if (item.ass_path && fs.existsSync(item.ass_path)) {
        try { fs.unlinkSync(item.ass_path); } catch {}
      }

      db.prepare('DELETE FROM voice_generations WHERE id = ?').run(req.params.id);
      res.json({ ok: true, deletedId: req.params.id });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.get('/api/voicebox/download/:id', (req, res) => {
    const item = db.prepare('SELECT * FROM voice_generations WHERE id = ?').get(req.params.id) as any;
    if (!item) {
      res.status(404).json({ error: 'Voice generation not found.' });
      return;
    }

    const format = req.query.format === 'mp3' ? 'mp3' : 'wav';
    const filePath = format === 'mp3' ? item.audio_path.replace(/\.wav$/i, '.mp3') : item.audio_path;

    if (!fs.existsSync(filePath)) {
      res.status(404).json({ error: 'Audio file is no longer available on disk.' });
      return;
    }

    const safeTitle = (item.title || 'voiceover')
      .replace(/[^a-zA-Z0-9_-]/g, '_')
      .slice(0, 40);
    const downloadFilename = `CreatorOS_Voicebox_${safeTitle}_${item.voice_id}.${format}`;

    res.download(filePath, downloadFilename);
  });

  // Media Search & Licensed Asset Acquisition
  app.get('/api/media/search', async (req, res) => {
    try {
      const q = String(req.query.q || '').trim();
      const aspectRatio = req.query.aspectRatio === '9:16' ? '9:16' : '16:9';
      if (!q) {
        res.status(400).json({ error: 'Search query q is required.' });
        return;
      }
      const candidates = await searchLicensedMedia(q, aspectRatio, 12);
      res.json({ candidates });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.post('/api/media/acquire', async (req, res) => {
    try {
      const { projectId, sceneIndex, candidate } = req.body || {};
      if (!candidate?.downloadUrl) {
        res.status(400).json({ error: 'Valid media candidate with downloadUrl is required.' });
        return;
      }
      const asset = await downloadAndVerifyMediaAsset({
        projectId,
        sceneIndex,
        candidate,
      });

      // Attach to storyboard scene if projectId and sceneIndex provided
      if (projectId && sceneIndex !== undefined) {
        const stb = db
          .prepare('SELECT * FROM storyboards WHERE project_id = ? ORDER BY created_at DESC LIMIT 1')
          .get(projectId) as any;
        if (stb) {
          const scenes = JSON.parse(stb.scenes_json || '[]');
          const target = scenes.find((s: any) => Number(s.sceneIndex) === Number(sceneIndex));
          if (target) {
            target.assetId = asset.id;
            target.assetUrl = asset.publicUrl;
            target.assetLicense = `${candidate.provider} (${candidate.licenseType})`;
            db.prepare('UPDATE storyboards SET scenes_json = ?, updated_at = ? WHERE id = ?').run(
              JSON.stringify(scenes),
              new Date().toISOString(),
              stb.id
            );
          }
        }
      }

      res.json({ ok: true, asset });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.get('/api/media/assets', (_req, res) => {
    const assets = db.prepare('SELECT * FROM media_assets ORDER BY created_at DESC LIMIT 60').all();
    res.json({ assets });
  });

  // Credits & Quota Management
  app.get('/api/credits', (_req, res) => {
    res.json(getUserCredits());
  });

  // Animation Engines & Capabilities
  app.get('/api/animation/engines', (_req, res) => {
    res.json({ engines: listAvailableEngines() });
  });

  // Video Generation Tasks & MiniMax H3 Integration
  app.get('/api/video-tasks/project/:projectId', (req, res) => {
    const tasks = listProjectVideoTasks(req.params.projectId);
    res.json({ tasks });
  });

  app.get('/api/video-tasks/:taskId', (req, res) => {
    const task = db.prepare('SELECT * FROM video_generation_tasks WHERE id = ?').get(req.params.taskId);
    if (!task) {
      res.status(404).json({ error: 'Task not found.' });
      return;
    }
    res.json({ task });
  });

  app.post('/api/video/enhance-prompt', async (req, res) => {
    try {
      const { prompt, style } = req.body || {};
      if (!prompt || !String(prompt).trim()) {
        res.status(400).json({ error: 'Prompt is required.' });
        return;
      }
      const enhanced = await enhanceVideoPrompt(String(prompt).trim(), style);
      res.json({ enhancedPrompt: enhanced });
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  app.post('/api/video-tasks/create', async (req, res) => {
    try {
      const {
        projectId,
        sceneIndex,
        prompt,
        enhancePrompt = false,
        provider = 'minimax_h3',
        style = 'cinematic_ai',
        mode = 'text_to_video',
        firstFrameUrl,
        lastFrameUrl,
        referenceImageUrl,
        aspectRatio = '9:16',
        durationSec = 5,
        resolution = '720p',
      } = req.body || {};

      if (!projectId || !prompt) {
        res.status(400).json({ error: 'projectId and prompt are required.' });
        return;
      }

      const result = await generateSingleScene({
        projectId,
        sceneIndex: Number(sceneIndex ?? 0),
        engine: provider,
        style,
        prompt,
        enhancePrompt,
        firstFrameUrl,
        lastFrameUrl,
        referenceImageUrl,
        durationSec: Number(durationSec),
        aspectRatio,
        resolution,
      });

      res.status(201).json(result);
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  // Unified Research Studio (Web, YouTube, or Combined)
  app.post('/api/research/unified', async (req, res) => {
    try {
      const { topic, niche = 'tech', mode = 'combined', projectId } = req.body || {};
      if (!topic || !String(topic).trim()) {
        res.status(400).json({ error: 'Topic is required.' });
        return;
      }
      const result = await runUnifiedResearchStudio({
        topic: String(topic).trim(),
        niche,
        mode,
        projectId,
      });
      res.json(result);
    } catch (err: any) {
      res.status(400).json({ error: err.message });
    }
  });

  // YouTube Competitors direct search
  app.get('/api/research/youtube-competitors', async (req, res) => {
    try {
      const q = String(req.query.q || '').trim();
      if (!q) {
        res.status(400).json({ error: 'Query q is required.' });
        return;
      }
      const items = await searchYouTubeCompetitors(q, Number(req.query.limit || 8));
      res.json({ items });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  // Scene Editor: Scene Management & Reordering
  app.post('/api/storyboard/:id/reorder-scenes', (req, res) => {
    const stb = db.prepare('SELECT * FROM storyboards WHERE id = ?').get(req.params.id) as any;
    if (!stb) {
      res.status(404).json({ error: 'Storyboard not found.' });
      return;
    }
    const { scenes } = req.body || {};
    if (!Array.isArray(scenes)) {
      res.status(400).json({ error: 'scenes array is required.' });
      return;
    }
    const reindexed = scenes.map((s: any, idx: number) => ({
      ...s,
      sceneIndex: idx + 1,
    }));
    const totalDur = Number(
      reindexed.reduce((acc: number, s: any) => acc + Math.max(1.5, Number(s.durationSec || 5)), 0).toFixed(1)
    );
    const now = new Date().toISOString();
    db.prepare('UPDATE storyboards SET scenes_json = ?, total_duration_sec = ?, updated_at = ? WHERE id = ?').run(
      JSON.stringify(reindexed),
      totalDur,
      now,
      stb.id
    );
    res.json({ storyboard: db.prepare('SELECT * FROM storyboards WHERE id = ?').get(stb.id) });
  });

  // Thumbnail Studio Endpoints
  app.get('/api/thumbnails', (req, res) => {
    const projectId = req.query.projectId ? String(req.query.projectId) : null;
    const rows = projectId
      ? db.prepare('SELECT * FROM thumbnails WHERE project_id = ? ORDER BY variant_index ASC').all(projectId)
      : db.prepare('SELECT * FROM thumbnails ORDER BY created_at DESC LIMIT 40').all();
    res.json({ thumbnails: rows });
  });

  app.post('/api/thumbnails/:id/approve', (req, res) => {
    const thumb = db.prepare('SELECT * FROM thumbnails WHERE id = ?').get(req.params.id) as any;
    if (!thumb) {
      res.status(404).json({ error: 'Thumbnail not found.' });
      return;
    }
    db.prepare('UPDATE thumbnails SET is_approved = 0 WHERE project_id = ?').run(thumb.project_id);
    db.prepare('UPDATE thumbnails SET is_approved = 1 WHERE id = ?').run(thumb.id);
    res.json({ ok: true, thumbnails: db.prepare('SELECT * FROM thumbnails WHERE project_id = ? ORDER BY variant_index ASC').all(thumb.project_id) });
  });

  app.post('/api/projects/:id/thumbnails/generate', async (req, res) => {
    try {
      const { customTitle, customSubtitle, backdropAssetId } = req.body || {};
      const result = await runThumbnailAgent({
        projectId: req.params.id,
        customTitle,
        customSubtitle,
        backdropAssetId,
      });
      res.json({ ok: true, ...result });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.post('/api/projects/:id/voice/generate', async (req, res) => {
    try {
      const { voiceName, voiceStyle, language, expressiveMode } = req.body || {};
      const result = await runVoiceAgent({
        projectId: req.params.id,
        voiceName,
        voiceStyle,
        language,
        expressiveMode,
      });
      res.json({ ok: true, ...result });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  // ---------------------------------------------------------------------------
  // OpenMontage Official Video Production & Quality Control Endpoints
  // ---------------------------------------------------------------------------

  app.get('/api/openmontage/status', async (_req, res) => {
    try {
      const status = await checkOpenMontageHealth();
      res.json(status);
    } catch (err) {
      res.status(500).json({ ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.get('/api/openmontage/pipelines', (_req, res) => {
    res.json({
      ok: true,
      pipelines: OPENMONTAGE_PIPELINES,
      totalCount: OPENMONTAGE_PIPELINES.length,
    });
  });

  app.post('/api/openmontage/assemble', async (req, res) => {
    try {
      const { projectId, styleKey, burnSubtitles, customResolution } = req.body || {};
      if (!projectId) {
        res.status(400).json({ error: 'projectId is required for OpenMontage video assembly.' });
        return;
      }
      const result = await assembleOpenMontageVideo({
        projectId,
        styleKey,
        burnSubtitles,
        customResolution,
      });

      // Automatically run Quality Control Agent on newly rendered MP4
      let qcReport: any = null;
      try {
        qcReport = await runQualityControlAgent({ projectId });
      } catch (qcErr) {
        console.warn('QC Agent post-assembly notice:', qcErr);
      }

      res.json({ ...result, qcReport });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.post('/api/openmontage/scene/regenerate', async (req, res) => {
    try {
      const { projectId, sceneIndex, effect, styleKey, newAssetId } = req.body || {};
      if (!projectId || sceneIndex === undefined) {
        res.status(400).json({ error: 'projectId and sceneIndex are required.' });
        return;
      }
      const result = await regenerateSceneSegment({
        projectId,
        sceneIndex: Number(sceneIndex),
        effect,
        styleKey,
        newAssetId,
      });
      res.json(result);
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.post('/api/openmontage/projects/:id/qc', async (req, res) => {
    try {
      const result = await runQualityControlAgent({ projectId: req.params.id });
      res.json({ ok: true, result });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.post('/api/projects/:id/approve', (req, res) => {
    const project = db.prepare('SELECT * FROM video_projects WHERE id = ?').get(req.params.id) as any;
    if (!project) {
      res.status(404).json({ error: 'Video project not found.' });
      return;
    }
    const now = new Date().toISOString();
    db.prepare(`
      UPDATE video_projects
      SET approval_status = 'approved', approved_at = ?, approved_by = 'usr_primary', rejection_reason = NULL, updated_at = ?
      WHERE id = ?
    `).run(now, now, project.id);

    updateProjectStageState(project.id, 'USER_APPROVAL', 'completed', 'Approved by Producer');
    res.json({ ok: true, approvalStatus: 'approved' });
  });

  app.post('/api/projects/:id/reject', (req, res) => {
    const project = db.prepare('SELECT * FROM video_projects WHERE id = ?').get(req.params.id) as any;
    if (!project) {
      res.status(404).json({ error: 'Video project not found.' });
      return;
    }
    const { reason = 'Rejected by producer for revision' } = req.body || {};
    const now = new Date().toISOString();
    db.prepare(`
      UPDATE video_projects
      SET approval_status = 'rejected', rejection_reason = ?, updated_at = ?
      WHERE id = ?
    `).run(reason, now, project.id);

    updateProjectStageState(project.id, 'USER_APPROVAL', 'failed', `Rejected: ${reason}`);
    res.json({ ok: true, approvalStatus: 'rejected', reason });
  });

  // SEO Metadata Update Endpoint
  app.put('/api/seo/:publishedId', (req, res) => {
    const rec = db.prepare('SELECT * FROM published_videos WHERE id = ?').get(req.params.publishedId) as any;
    if (!rec) {
      res.status(404).json({ error: 'SEO metadata record not found.' });
      return;
    }
    const { title, description, tags, categoryId } = req.body || {};
    db.prepare(`
      UPDATE published_videos
      SET title = ?, description = ?, tags_json = ?, category_id = ?, updated_at = ?
      WHERE id = ?
    `).run(
      title !== undefined ? String(title) : rec.title,
      description !== undefined ? String(description) : rec.description,
      Array.isArray(tags) ? JSON.stringify(tags) : rec.tags_json,
      categoryId !== undefined ? String(categoryId) : rec.category_id,
      new Date().toISOString(),
      rec.id
    );
    res.json({ publishedRecord: db.prepare('SELECT * FROM published_videos WHERE id = ?').get(rec.id) });
  });

  // YouTube OAuth 2.0 & Channel Endpoints
  app.get('/api/youtube/auth/url', (req, res) => {
    const callbackUrl = getYouTubeRedirectUri(req);
    const oauthUris = getAuthorizedOAuthUris(req);
    const { clientId, isConfigured, warning, missing } = getOAuthCredentials();

    if (!isConfigured) {
      res.status(400).json({
        configured: false,
        error:
          warning ||
          'Google / YouTube OAuth 2.0 Web Application credentials are not configured in server environment variables. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET (or YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET).',
        warning,
        missing,
        callbackUrl,
        clientId,
        devCallbackUrl: oauthUris.devUri,
        previewCallbackUrl: oauthUris.previewUri,
        authorizedUris: oauthUris.allUris,
      });
      return;
    }
    try {
      const state = encodeOAuthState(callbackUrl);
      const url = buildYouTubeAuthUrl(state, callbackUrl);
      res.json({
        configured: true,
        url,
        callbackUrl,
        clientId,
        warning,
        devCallbackUrl: oauthUris.devUri,
        previewCallbackUrl: oauthUris.previewUri,
        authorizedUris: oauthUris.allUris,
      });
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  const oauthCallbackHandler = async (req: express.Request, res: express.Response) => {
    const code = req.query.code ? String(req.query.code) : '';
    const stateStr = req.query.state ? String(req.query.state) : '';
    const oauthErr = req.query.error ? String(req.query.error) : '';
    const errorDesc = req.query.error_description ? String(req.query.error_description) : '';

    if (oauthErr) {
      let explanation = 'Google authorization was not completed.';
      if (oauthErr === 'access_denied') {
        explanation = 'The permission request was cancelled or denied in the Google consent screen.';
      } else if (oauthErr === 'invalid_client') {
        explanation = 'Google Cloud could not locate the OAuth Client ID. Verify that GOOGLE_CLIENT_ID matches an active Web Application client in Google Cloud Console.';
      } else if (oauthErr === 'redirect_uri_mismatch') {
        explanation = 'The redirect URI sent to Google does not match the Authorized redirect URIs configured in Google Cloud Console.';
      }

      res.status(400).send(`
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>YouTube Authorization Error — CreatorOS AI</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0B0F17; color: #F8FAFC; padding: 32px 16px; display: flex; align-items: center; justify-content: center; min-height: 80vh; margin: 0; }
            .card { background: #111722; border: 1px solid #1E293B; border-radius: 12px; padding: 32px; max-width: 520px; width: 100%; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            h2 { color: #F43F5E; margin-top: 0; font-size: 20px; }
            p { color: #94A3B8; font-size: 14px; line-height: 1.6; }
            .badge { display: inline-block; background: #2D1515; color: #FB7185; padding: 4px 10px; border-radius: 6px; font-family: monospace; font-size: 12px; margin-bottom: 14px; }
            .desc { font-family: monospace; font-size: 12px; background: #05070B; padding: 12px; border-radius: 6px; color: #CBD5E1; word-break: break-all; margin: 12px 0; }
            .btn { display: inline-block; background: #E11D48; color: #fff; text-decoration: none; padding: 10px 18px; border-radius: 6px; font-weight: 600; font-size: 13px; cursor: pointer; border: none; }
          </style>
        </head>
        <body>
          <div class="card">
            <span class="badge">${oauthErr}</span>
            <h2>Authorization Not Completed</h2>
            <p>${explanation}</p>
            ${errorDesc ? `<div class="desc">${errorDesc}</div>` : ''}
            <div style="margin-top: 24px;">
              <button class="btn" onclick="if (window.opener) { window.opener.postMessage({ type: 'YOUTUBE_OAUTH_ERROR', error: ${JSON.stringify(oauthErr + (errorDesc ? ': ' + errorDesc : ''))} }, '*'); window.close(); } else { window.location.href = '/studio?section=youtube_channels'; }">Return to Studio</button>
            </div>
          </div>
          <script>
            if (window.opener) {
              window.opener.postMessage({ type: 'YOUTUBE_OAUTH_ERROR', error: ${JSON.stringify(oauthErr + (errorDesc ? ': ' + errorDesc : ''))} }, '*');
            }
          </script>
        </body>
        </html>
      `);
      return;
    }

    if (!code) {
      res.status(400).send(`
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"><title>Error</title><style>body { font-family: sans-serif; background: #0B0F17; color: #F8FAFC; padding: 40px; }</style></head>
        <body><h3>Missing OAuth Authorization Code</h3><p>Google OAuth did not provide a verification code.</p></body>
        </html>
      `);
      return;
    }

    try {
      const decoded = decodeOAuthState(stateStr);
      const redirectUri = decoded.redirectUri || getYouTubeRedirectUri(req);
      const channel = await handleYouTubeOAuthCallback(code, 'usr_primary', redirectUri);
      res.send(`
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>YouTube Connected — CreatorOS AI</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0B0F17; color: #F8FAFC; padding: 32px 16px; display: flex; align-items: center; justify-content: center; min-height: 80vh; margin: 0; }
            .card { background: #111722; border: 1px solid #1E293B; border-radius: 12px; padding: 32px; max-width: 520px; width: 100%; text-align: center; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            h2 { color: #10B981; margin-top: 0; font-size: 20px; }
            p { color: #94A3B8; font-size: 14px; line-height: 1.6; }
            .tag { display: inline-block; background: #064E3B; color: #A7F3D0; padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; margin-bottom: 12px; }
          </style>
        </head>
        <body>
          <div class="card">
            <span class="tag">Verified Channel</span>
            <h2>${channel.title} Connected!</h2>
            <p>${channel.handle || channel.youtubeChannelId} has been successfully authorized.</p>
            <p>Access and refresh tokens were encrypted with AES-256-GCM in the secure server token vault.</p>
            <p style="font-size: 12px; color: #64748B;">Closing window automatically...</p>
          </div>
          <script>
            if (window.opener) {
              window.opener.postMessage({ type: 'YOUTUBE_OAUTH_SUCCESS', channel: ${JSON.stringify(channel)} }, '*');
              setTimeout(() => {
                try { window.close(); } catch (e) {}
              }, 1200);
            } else {
              setTimeout(() => { window.location.href = '/studio?section=youtube_channels'; }, 1500);
            }
          </script>
        </body>
        </html>
      `);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      res.status(500).send(`
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>OAuth Token Exchange Failed — CreatorOS AI</title>
          <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0B0F17; color: #F8FAFC; padding: 32px 16px; display: flex; align-items: center; justify-content: center; min-height: 80vh; margin: 0; }
            .card { background: #111722; border: 1px solid #1E293B; border-radius: 12px; padding: 32px; max-width: 520px; width: 100%; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
            h2 { color: #F43F5E; margin-top: 0; font-size: 20px; }
            p { color: #94A3B8; font-size: 14px; line-height: 1.6; }
            .err { background: #1E1218; border: 1px solid #3B1D28; color: #FDA4AF; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 12px; word-break: break-all; margin: 12px 0; }
            .btn { background: #334155; color: #fff; border: none; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 13px; }
          </style>
        </head>
        <body>
          <div class="card">
            <h2>OAuth Exchange Failed</h2>
            <p>The authorization code could not be exchanged with Google for access tokens:</p>
            <div class="err">${msg}</div>
            <div style="margin-top: 20px;">
              <button class="btn" onclick="if (window.opener) { window.opener.postMessage({ type: 'YOUTUBE_OAUTH_ERROR', error: ${JSON.stringify(msg)} }, '*'); window.close(); } else { window.location.href = '/studio?section=youtube_channels'; }">Return to Studio</button>
            </div>
          </div>
          <script>
            if (window.opener) {
              window.opener.postMessage({ type: 'YOUTUBE_OAUTH_ERROR', error: ${JSON.stringify(msg)} }, '*');
            }
          </script>
        </body>
        </html>
      `);
    }
  };

  app.get(['/api/youtube/auth/callback', '/api/youtube/auth/callback/'], oauthCallbackHandler);

  app.get('/api/youtube/channels', (_req, res) => {
    const channels = db
      .prepare(
        `SELECT c.*, o.status as oauth_status, o.last_refreshed_at, o.scope,
                (o.encrypted_refresh_token IS NOT NULL AND o.encrypted_refresh_token != '') as has_refresh_token
         FROM youtube_channels c
         LEFT JOIN oauth_connections o ON o.channel_id = c.id
         ORDER BY c.is_primary DESC, c.connected_at DESC`
      )
      .all();
    res.json({ channels });
  });

  app.post('/api/youtube/channels/:id/primary', (req, res) => {
    db.prepare('UPDATE youtube_channels SET is_primary = 0').run();
    db.prepare('UPDATE youtube_channels SET is_primary = 1 WHERE id = ?').run(req.params.id);
    res.json({ ok: true });
  });

  app.delete('/api/youtube/channels/:id', (req, res) => {
    db.prepare('DELETE FROM oauth_connections WHERE channel_id = ?').run(req.params.id);
    db.prepare('DELETE FROM youtube_channels WHERE id = ?').run(req.params.id);
    res.json({ ok: true });
  });

  // Publishing History & Upload Endpoints
  app.get('/api/publishing/history', (_req, res) => {
    const items = db
      .prepare(
        `SELECT pv.*, vp.topic, vp.format, vp.language, vp.approval_status, vp.final_video_url,
                yc.title as channel_title, yc.handle as channel_handle
         FROM published_videos pv
         LEFT JOIN video_projects vp ON vp.id = pv.project_id
         LEFT JOIN youtube_channels yc ON yc.id = pv.channel_id
         ORDER BY pv.updated_at DESC`
      )
      .all();
    res.json({ publishedVideos: items });
  });

  app.post('/api/publishing/upload', async (req, res) => {
    try {
      const { projectId, channelId, privacyStatus = 'private', publishAt, title, description, tags } =
        req.body || {};
      if (!projectId) {
        res.status(400).json({ error: 'projectId is required.' });
        return;
      }
      const result = await runPublishingAgent({
        projectId,
        channelId,
        privacyStatus,
        publishAt: publishAt || null,
        title,
        description,
        tags,
      });
      res.json({ ok: true, result });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  // Analytics Endpoints
  app.get('/api/analytics/summary', (_req, res) => {
    const records = db
      .prepare(
        `SELECT a.*, pv.title as video_title, yc.title as channel_title
         FROM analytics a
         LEFT JOIN published_videos pv ON pv.id = a.published_video_id
         LEFT JOIN youtube_channels yc ON yc.id = a.channel_id
         ORDER BY a.fetched_at DESC LIMIT 60`
      )
      .all();
    const channels = db.prepare('SELECT * FROM youtube_channels ORDER BY is_primary DESC').all();
    res.json({ records, channels });
  });

  app.post('/api/analytics/sync', async (req, res) => {
    try {
      const { channelId } = req.body || {};
      const report = await runAnalyticsAgent({ channelId });
      res.json({ ok: true, report });
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  // Automation Scheduler Endpoints
  app.get('/api/schedules', (_req, res) => {
    const schedules = db.prepare('SELECT * FROM publishing_schedules ORDER BY created_at DESC').all();
    res.json({ schedules });
  });

  app.post('/api/schedules', (req, res) => {
    const {
      name,
      scheduleType = 'daily_short',
      cronExpression = '0 9 * * *',
      scheduledFor,
      projectId,
      channelId,
      niche = 'tech',
      format = 'short',
      language = 'en',
      privacyStatus = 'private',
      autoPublish = false,
    } = req.body || {};

    if (!name || !String(name).trim()) {
      res.status(400).json({ error: 'Schedule name is required.' });
      return;
    }

    const id = newId('sch');
    const now = new Date().toISOString();
    const nextRun =
      scheduleType === 'one_time' && scheduledFor
        ? new Date(scheduledFor).toISOString()
        : computeNextRunIso(scheduleType, cronExpression);

    db.prepare(`
      INSERT INTO publishing_schedules (
        id, user_id, channel_id, project_id, name, schedule_type,
        cron_expression, scheduled_for, niche, format, language,
        privacy_status, auto_publish, enabled, next_run_at, created_at
      ) VALUES (?, 'usr_primary', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
    `).run(
      id,
      channelId || null,
      projectId || null,
      String(name).trim(),
      scheduleType,
      cronExpression,
      scheduledFor || null,
      niche,
      format === 'long' ? 'long' : 'short',
      language === 'ur' ? 'ur' : 'en',
      privacyStatus,
      autoPublish ? 1 : 0,
      nextRun,
      now
    );

    res.status(201).json({
      schedule: db.prepare('SELECT * FROM publishing_schedules WHERE id = ?').get(id),
    });
  });

  app.patch('/api/schedules/:id', (req, res) => {
    const existing = db.prepare('SELECT * FROM publishing_schedules WHERE id = ?').get(req.params.id) as any;
    if (!existing) {
      res.status(404).json({ error: 'Schedule not found.' });
      return;
    }
    const { enabled, autoPublish } = req.body || {};
    db.prepare('UPDATE publishing_schedules SET enabled = ?, auto_publish = ? WHERE id = ?').run(
      enabled !== undefined ? (enabled ? 1 : 0) : existing.enabled,
      autoPublish !== undefined ? (autoPublish ? 1 : 0) : existing.auto_publish,
      existing.id
    );
    res.json({ schedule: db.prepare('SELECT * FROM publishing_schedules WHERE id = ?').get(existing.id) });
  });

  app.post('/api/schedules/:id/run', async (req, res) => {
    try {
      const outcome = await executeScheduleRule(req.params.id);
      res.json(outcome);
    } catch (err) {
      res.status(400).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.delete('/api/schedules/:id', (req, res) => {
    db.prepare('DELETE FROM publishing_schedules WHERE id = ?').run(req.params.id);
    res.json({ ok: true });
  });

  // Jobs & Agent Logs Endpoints
  app.get('/api/jobs', (_req, res) => {
    const jobs = db
      .prepare(
        `SELECT j.*, vp.title as project_title, vp.topic as project_topic
         FROM production_jobs j
         LEFT JOIN video_projects vp ON vp.id = j.project_id
         ORDER BY j.created_at DESC LIMIT 60`
      )
      .all();
    res.json({ jobs });
  });

  app.post('/api/jobs/:id/retry', (req, res) => {
    const job = db.prepare('SELECT * FROM production_jobs WHERE id = ?').get(req.params.id) as any;
    if (!job) {
      res.status(404).json({ error: 'Job not found.' });
      return;
    }
    const payload = job.payload_json ? JSON.parse(job.payload_json) : {};
    const next = enqueueProductionJob({
      projectId: job.project_id,
      stage: job.stage,
      payload,
    });
    res.json({ ok: true, job: next });
  });

  app.get('/api/agents/logs', (_req, res) => {
    const logs = db
      .prepare('SELECT * FROM agent_execution_logs ORDER BY created_at DESC LIMIT 100')
      .all();
    res.json({ logs });
  });

  app.get('/api/agents/status', (_req, res) => {
    const agentDefs = [
      { id: 'agent_research', number: 1, name: 'Research Agent' },
      { id: 'agent_strategy', number: 2, name: 'Strategy Agent' },
      { id: 'agent_scriptwriter', number: 3, name: 'Scriptwriting Agent' },
      { id: 'agent_factcheck', number: 4, name: 'Fact-Checking Agent' },
      { id: 'agent_storyboard', number: 5, name: 'Storyboard Agent' },
      { id: 'agent_voice', number: 6, name: 'Voice Agent' },
      { id: 'agent_video_producer', number: 7, name: 'Video Production Agent' },
      { id: 'agent_thumbnail', number: 8, name: 'Thumbnail Agent' },
      { id: 'agent_seo', number: 9, name: 'SEO Agent' },
      { id: 'agent_publishing', number: 10, name: 'Publishing Agent' },
      { id: 'agent_analytics', number: 11, name: 'Analytics Agent' },
    ];

    const summaryRows = db
      .prepare(
        `SELECT agent_id,
                COUNT(*) as total_runs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_runs,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
                MAX(created_at) as last_run_at
         FROM agent_execution_logs
         GROUP BY agent_id`
      )
      .all() as any[];

    const byAgent = new Map(summaryRows.map((r) => [r.agent_id, r]));
    const latestStmt = db.prepare(
      'SELECT status, model_used, duration_ms, output_summary, error_message, created_at FROM agent_execution_logs WHERE agent_id = ? ORDER BY created_at DESC LIMIT 1'
    );

    const agents = agentDefs.map((def) => {
      const agg = byAgent.get(def.id);
      const latest = latestStmt.get(def.id) as any;
      return {
        ...def,
        status: latest?.status || 'idle',
        totalRuns: Number(agg?.total_runs || 0),
        completedRuns: Number(agg?.completed_runs || 0),
        failedRuns: Number(agg?.failed_runs || 0),
        lastRunAt: agg?.last_run_at || null,
        lastModel: latest?.model_used || null,
        lastDurationMs: latest?.duration_ms ?? null,
        lastSummary: latest?.output_summary || latest?.error_message || null,
      };
    });

    res.json({ agents });
  });

  // GitHub Integrations Manager Endpoints
  app.get('/api/github/integrations', (_req, res) => {
    const integrations = db
      .prepare('SELECT * FROM github_integrations ORDER BY created_at ASC')
      .all();
    res.json({ integrations });
  });

  app.post('/api/github/verify', async (_req, res) => {
    const integrations = db.prepare('SELECT * FROM github_integrations').all() as any[];
    const now = new Date().toISOString();
    const results = [];

    for (const item of integrations) {
      const modules = JSON.parse(item.integrated_modules_json || '[]') as Array<{
        targetModule: string;
        sourceFile: string;
        purpose: string;
      }>;
      let allPresent = true;
      for (const m of modules) {
        const targetPath = path.resolve(process.cwd(), m.targetModule);
        if (!fs.existsSync(targetPath)) {
          allPresent = false;
        }
      }
      const status = allPresent
        ? item.license === 'GPL-3.0'
          ? 'cleanroom_adapted'
          : 'integrated'
        : 'needs_sync';
      db.prepare('UPDATE github_integrations SET integration_status = ?, last_verified_at = ? WHERE id = ?').run(
        status,
        now,
        item.id
      );
      results.push({ id: item.id, repo_slug: item.repo_slug, status, verifiedAt: now });
    }

    res.json({ ok: true, results, integrations: db.prepare('SELECT * FROM github_integrations').all() });
  });

  // Settings Endpoints (User preferences, safety gates, never API keys)
  app.get('/api/settings', (_req, res) => {
    const user = db.prepare('SELECT * FROM users WHERE id = ?').get('usr_primary');
    res.json({ settings: user });
  });

  app.put('/api/settings', (req, res) => {
    const user = db.prepare('SELECT * FROM users WHERE id = ?').get('usr_primary') as any;
    const {
      name,
      defaultLanguage,
      defaultNiche,
      autoPublishEnabled,
      requireFactCheck,
      requireLicenseCheck,
    } = req.body || {};

    db.prepare(`
      UPDATE users SET
        name = ?,
        default_language = ?,
        default_niche = ?,
        auto_publish_enabled = ?,
        require_fact_check = ?,
        require_license_check = ?,
        updated_at = ?
      WHERE id = 'usr_primary'
    `).run(
      name !== undefined ? String(name) : user.name,
      defaultLanguage === 'ur' ? 'ur' : defaultLanguage === 'en' ? 'en' : user.default_language,
      defaultNiche !== undefined ? String(defaultNiche) : user.default_niche,
      autoPublishEnabled !== undefined ? (autoPublishEnabled ? 1 : 0) : user.auto_publish_enabled,
      requireFactCheck !== undefined ? (requireFactCheck ? 1 : 0) : user.require_fact_check,
      requireLicenseCheck !== undefined ? (requireLicenseCheck ? 1 : 0) : user.require_license_check,
      new Date().toISOString()
    );

    res.json({ settings: db.prepare('SELECT * FROM users WHERE id = ?').get('usr_primary') });
  });

  // Start background job queue and server-side cron scheduler
  startBackgroundJobWorker();
  startAutomationScheduler();

  // Reject unmatched /api/* routes with structured JSON 404 before SPA catch-all
  app.use('/api', (_req, res) => {
    res.status(404).json({ error: 'API endpoint not found.' });
  });

  const httpServer = http.createServer(app);

  // Frontend Vite middleware (dev) or static dist serving (production)
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const isHmrDisabled = process.env.DISABLE_HMR === 'true';
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        allowedHosts: true,
        ws: isHmrDisabled ? false : { server: httpServer },
        hmr: isHmrDisabled ? false : undefined,
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distDir));
    app.use((_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  // Dev server must run on port 3000 behind Nginx reverse proxy (which listens on 8080 and proxies to 3000)
  const portArgIdx = process.argv.indexOf('--port');
  const cliPort = portArgIdx !== -1 && process.argv[portArgIdx + 1] ? Number(process.argv[portArgIdx + 1]) : null;
  const envPort = process.env.PORT && process.env.PORT !== '8080' ? Number(process.env.PORT) : 3000;
  const PORT = cliPort || Number(process.env.DEFAULT_APP_PORT || 3000) || envPort;

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`CreatorOS AI Studio server running on http://0.0.0.0:${PORT}`);
  });
}

createServer().catch((err) => {
  console.error('Fatal server startup error:', err);
  process.exit(1);
});

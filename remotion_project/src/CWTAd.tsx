import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  Sequence,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';

type Scene = {
  imagePath: string;
  narration?: string;
  startFrame: number;
  durationFrames: number;
};

type Subtitle = {
  text: string;
  startFrame: number;
  endFrame: number;
};

type CWTAdProps = {
  scenes: Scene[];
  audioPath: string;
  subtitles: Subtitle[];
};

export const CWTAd: React.FC<CWTAdProps> = ({scenes = [], audioPath = '', subtitles = []}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  return (
    <AbsoluteFill style={{backgroundColor: '#000000'}}>
      {audioPath ? <Audio src={audioPath} /> : null}

      {scenes.map((scene, i) => {
        return (
          <Sequence key={i} from={scene.startFrame} durationInFrames={scene.durationFrames}>
            <AbsoluteFill>
              <Img
                src={scene.imagePath}
                style={{width: '100%', height: '100%', objectFit: 'cover'}}
              />

              <AbsoluteFill style={{background: 'linear-gradient(transparent, rgba(0,0,0,0.6))'}}>
                <div
                  style={{
                    position: 'absolute',
                    bottom: '12%',
                    width: '100%',
                    display: 'flex',
                    justifyContent: 'center',
                    padding: '0 20px',
                    boxSizing: 'border-box',
                    pointerEvents: 'none',
                  }}
                >
                  <div style={{maxWidth: 920, textAlign: 'center'}}>
                    {subtitles
                      .filter((s) => frame >= s.startFrame && frame <= s.endFrame)
                      .map((s, idx) => (
                        <div
                          key={idx}
                          style={{
                            color: 'white',
                            fontSize: 42,
                            lineHeight: '1.1',
                            textShadow: '0 2px 8px rgba(0,0,0,0.8)',
                            fontFamily: 'Inter, Arial, sans-serif',
                          }}
                        >
                          {s.text}
                        </div>
                      ))}
                  </div>
                </div>
              </AbsoluteFill>
            </AbsoluteFill>
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

export default CWTAd;
import React from "react";
import {
  AbsoluteFill,
  Audio,
  Img,
  Sequence,
  useCurrentFrame,
  staticFile,
} from "remotion";

/* ------------------------------------------------------------------ *
 * Props interface
 * ------------------------------------------------------------------ */

interface SceneProps {
  imagePath: string;
  narration: string;
  startFrame: number;
  durationFrames: number;
}

interface SubtitleProps {
  text: string;
  startFrame: number;
  endFrame: number;
}

export interface CWTAdProps {
  scenes: SceneProps[];
  audioPath: string;
  subtitles: SubtitleProps[];
}

/* ------------------------------------------------------------------ *
 * Subtitle overlay component
 * ------------------------------------------------------------------ */

const SubtitleOverlay: React.FC<{
  subtitles: SubtitleProps[];
  sceneStart: number;
  sceneDuration: number;
}> = ({ subtitles, sceneStart, sceneDuration }) => {
  const frame = useCurrentFrame();
  const globalFrame = sceneStart + frame;

  const activeSubtitles = subtitles.filter(
    (sub) => globalFrame >= sub.startFrame && globalFrame < sub.endFrame
  );

  if (activeSubtitles.length === 0) return null;

  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(transparent, rgba(0,0,0,0.6))",
        justifyContent: "flex-end",
        alignItems: "center",
        padding: "0 40px 120px 40px",
      }}
    >
      {activeSubtitles.map((sub, idx) => (
        <div
          key={idx}
          style={{
            color: "#FFFFFF",
            fontSize: 42,
            fontWeight: 700,
            textAlign: "center",
            lineHeight: 1.3,
            textShadow: "0 2px 8px rgba(0,0,0,0.8)",
            maxWidth: "90%",
          }}
        >
          {sub.text}
        </div>
      ))}
    </AbsoluteFill>
  );
};

/* ------------------------------------------------------------------ *
 * Main composition
 * ------------------------------------------------------------------ */

export const CWTAd: React.FC<CWTAdProps> = ({
  scenes,
  audioPath,
  subtitles,
}) => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#000000" }}>
      {/* Voice-over audio track */}
      <Audio src={staticFile(audioPath)} />

      {/* Scene sequences */}
      {scenes.map((scene, i) => (
        <Sequence
          key={i}
          from={scene.startFrame}
          durationInFrames={scene.durationFrames}
        >
          {/* Background scene image */}
          <Img
            src={staticFile(scene.imagePath)}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
            }}
          />

          {/* Subtitle overlay (bottom 20%, white text) */}
          <SubtitleOverlay
            subtitles={subtitles}
            sceneStart={scene.startFrame}
            sceneDuration={scene.durationFrames}
          />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

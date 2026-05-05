import React from "react";
import {
  AbsoluteFill,
  Audio,
  Img,
  Sequence,
  useCurrentFrame,
} from "remotion";

interface SceneProps {
  imagePath: string;
  narration?: string;
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

const SubtitleOverlay: React.FC<{
  subtitles: SubtitleProps[];
  sceneStart: number;
}> = ({ subtitles, sceneStart }) => {
  const frame = useCurrentFrame();
  const globalFrame = sceneStart + frame;

  const activeSubtitles = subtitles.filter(
    (sub) => globalFrame >= sub.startFrame && globalFrame < sub.endFrame
  );

  if (activeSubtitles.length === 0) {
    return null;
  }

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

export const CWTAd: React.FC<CWTAdProps> = ({ scenes, audioPath, subtitles }) => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#000000" }}>
      {audioPath ? <Audio src={audioPath} /> : null}

      {scenes.map((scene, i) => (
        <Sequence
          key={i}
          from={scene.startFrame}
          durationInFrames={scene.durationFrames}
        >
          <Img
            src={scene.imagePath}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
            }}
          />

          <SubtitleOverlay subtitles={subtitles} sceneStart={scene.startFrame} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};

export default CWTAd;
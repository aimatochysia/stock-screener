import React from "react";

const ProgressBar = ({ progress, total, message = "Loading..." }) => {
  const percentage = total > 0 ? Math.round((progress / total) * 100) : 0;

  return (
    <div className="progress-container">
      <div className="progress-message">{message}</div>
      <div className="progress-bar">
        <div
          className="progress-fill"
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
      <div className="progress-text">
        {progress} / {total} ({percentage}%)
      </div>
    </div>
  );
};

export default ProgressBar;

# PowerShell script to safely push the repository with LFS
# Ensure we are in the project root
Set-Location "C:\\Users\\AYUSH\\Documents\\MLChallenge"

# Increase HTTP post buffer and LFS settings for large uploads
git config http.postBuffer 1073741824
git config http.maxRequestBuffer 1073741824
git config http.lowSpeedLimit 0
git config http.lowSpeedTime 999999
git config lfs.concurrenttransfers 5

# Remove stale lock file if it exists
if (Test-Path .git\\index.lock) {
    Write-Host "Removing stale index.lock..."
    Remove-Item -Force .git\\index.lock
}

# Push all LFS objects to the remote
Write-Host "Pushing LFS objects..."
git lfs push --all origin master

# Push commits to the remote
Write-Host "Pushing commits..."
git push origin master

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Assert-NoMatches {
	param(
		[Parameter(Mandatory = $true)][string]$Path,
		[Parameter(Mandatory = $true)][string]$Pattern,
		[Parameter(Mandatory = $true)][string]$Name
	)

	$content = Get-Content -Path $Path -Raw -Encoding UTF8
	if ($content -match $Pattern) {
		throw "[CHECK FAILED] $Name"
	}
	Write-Host "[OK] $Name"
}

function Assert-Match {
	param(
		[Parameter(Mandatory = $true)][string]$Path,
		[Parameter(Mandatory = $true)][string]$Pattern,
		[Parameter(Mandatory = $true)][string]$Name
	)

	$content = Get-Content -Path $Path -Raw -Encoding UTF8
	if ($content -notmatch $Pattern) {
		throw "[CHECK FAILED] $Name"
	}
	Write-Host "[OK] $Name"
}

function Assert-StubbedFunctions {
	param(
		[Parameter(Mandatory = $true)][string]$Path,
		[Parameter(Mandatory = $true)][string[]]$FunctionNames,
		[Parameter(Mandatory = $true)][string]$Prefix
	)

	$content = Get-Content -Path $Path -Raw -Encoding UTF8
	foreach ($name in $FunctionNames) {
		$escaped = [regex]::Escape($name)
		$pattern = "int\s+$escaped\s*\([^\)]*\)\s*\{[^\{\}]*return\s+0\s*;[^\{\}]*\}"
		if ($content -notmatch $pattern) {
			throw "[CHECK FAILED] $Prefix $name is not trimmed to return 0"
		}
	}
	Write-Host "[OK] $Prefix stubs verified: $($FunctionNames.Count)"
}

function Assert-EmptyKernelFunctions {
	param(
		[Parameter(Mandatory = $true)][string]$Path,
		[Parameter(Mandatory = $true)][string[]]$FunctionNames,
		[Parameter(Mandatory = $true)][string]$Prefix
	)

	$content = Get-Content -Path $Path -Raw -Encoding UTF8
	foreach ($name in $FunctionNames) {
		$escaped = [regex]::Escape($name)
		$pattern = "__global__\s+void\s+$escaped\s*\([\s\S]*?\)\s*\{\s*\}"
		if ($content -notmatch $pattern) {
			throw "[CHECK FAILED] $Prefix kernel $name is not empty"
		}
	}
	Write-Host "[OK] $Prefix empty kernels verified: $($FunctionNames.Count)"
}

Write-Host "=== Step 1/3: Run trim script ==="
python .\trim_lite.py

Write-Host "=== Step 2/3: Validate trim rules ==="
Assert-NoMatches -Path ".\cccc\cccc.vcxproj" -Pattern "predict\.cpp|predict\.h|cccc\.rc|auto_version\.exe" -Name "cccc.vcxproj removed predict/rc/auto_version"
Assert-Match -Path ".\cccc\MainProcess.cpp" -Pattern "MP_count_\s*=\s*1\s*;" -Name "MainProcess enforces single GPU"

$cudaBwd = @(
	'cuda_layer_norm_bwd',
	'cuda_rms_norm_bwd',
	'cuda_rope_bwd',
	'cuda_pixel_shuffle_bwd',
	'cuda_embed_bwd',
	'cuda_tile_bwd',
	'cuda_group_norm_affine_bwd',
	'cuda_reparam_bwd',
	'cuda_l1_bwd',
	'cuda_kl_lv_bwd',
	'cuda_upsample_nearest_bwd',
	'cuda_upsample_bilinear_bwd'
)

$hipBwd = @(
	'hip_layer_norm_bwd',
	'hip_rms_norm_bwd',
	'hip_rope_bwd',
	'hip_pixel_shuffle_bwd',
	'hip_embed_bwd',
	'hip_tile_bwd',
	'hip_group_norm_affine_bwd',
	'hip_reparam_bwd',
	'hip_l1_bwd',
	'hip_kl_lv_bwd',
	'hip_upsample_nearest_bwd',
	'hip_upsample_bilinear_bwd'
)

$cudaBwdKernels = @(
	'layer_norm_bwd_float_kernel',
	'rms_norm_bwd_float_kernel',
	'rope_bwd_float_kernel',
	'pixel_shuffle_bwd_kernel',
	'embed_bwd_float_kernel',
	'tile_bwd_float_kernel',
	'group_norm_affine_bwd_float',
	'reparam_bwd_float',
	'l1_bwd_float',
	'kl_lv_bwd_float',
	'upsample_nearest_bwd_float',
	'upsample_bilinear_bwd_float'
)

$hipBwdKernels = @(
	'layer_norm_bwd_float_kernel',
	'layer_norm_bwd_bf16_kernel',
	'rms_norm_bwd_float_kernel',
	'rms_norm_bwd_bf16_kernel',
	'rope_bwd_float_kernel',
	'rope_bwd_bf16_kernel',
	'pixel_shuffle_bwd_kernel',
	'pixel_shuffle16_bwd_kernel',
	'embed_bwd_float_kernel',
	'embed_bwd_bf16_kernel',
	'tile_bwd_float_kernel',
	'tile_bwd_bf16_kernel',
	'group_norm_affine_bwd_float',
	'group_norm_affine_bwd_bf16',
	'reparam_bwd_float',
	'reparam_bwd_bf16',
	'l1_bwd_float',
	'l1_bwd_bf16',
	'kl_lv_bwd_float',
	'kl_lv_bwd_bf16',
	'upsample_nearest_bwd_float',
	'upsample_nearest_bwd_bf16',
	'upsample_bilinear_bwd_float',
	'upsample_bilinear_bwd_bf16'
)

Assert-StubbedFunctions -Path ".\cccc-cuda\cuda_functions.cu" -FunctionNames $cudaBwd -Prefix "CUDA"
Assert-StubbedFunctions -Path ".\cccc-hip\hip_functions.cpp" -FunctionNames $hipBwd -Prefix "HIP"
Assert-EmptyKernelFunctions -Path ".\cccc-cuda\cuda_functions.cu" -FunctionNames $cudaBwdKernels -Prefix "CUDA"
Assert-EmptyKernelFunctions -Path ".\cccc-hip\hip_functions.cpp" -FunctionNames $hipBwdKernels -Prefix "HIP"

Write-Host "=== Step 3/3: Build verification (Release|x64) ==="
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
	throw "vswhere not found: $vswhere"
}

$msbuild = & $vswhere -latest -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe" | Select-Object -First 1
if (-not $msbuild) {
	throw "MSBuild not found via vswhere"
}

& $msbuild ".\cccc-lite.sln" "/p:Configuration=Release" "/p:Platform=x64" "/nologo" "/clp:ErrorsOnly;Summary"
if ($LASTEXITCODE -ne 0) {
	throw "MSBuild failed, exit code: $LASTEXITCODE"
}

Write-Host "=== Pipeline finished successfully ==="

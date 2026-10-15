function Test-LearningResponse {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Question,

        [ValidateSet("short", "long")]
        [string]$AnswerStyle = "short",

        [string]$BaseUrl = "http://localhost:5000",

        [Parameter(Mandatory = $true)]
        [string]$AccessToken
    )

    $body = @{
        question = $Question
        answer_style = $AnswerStyle
    } | ConvertTo-Json

    Invoke-RestMethod `
        -Method Post `
        -Uri "$BaseUrl/api/learning-response" `
        -Headers @{ Authorization = "Bearer $AccessToken" } `
        -ContentType "application/json" `
        -Body $body
}
